from __future__ import annotations

import argparse
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd
import requests
from requests import Response
from tqdm import tqdm

from config import ensure_directories, load_settings


POSTS_QUERY = """
query GetPosts($first: Int!, $after: String, $postedAfter: DateTime, $postedBefore: DateTime) {
  posts(first: $first, after: $after, postedAfter: $postedAfter, postedBefore: $postedBefore) {
    edges {
      cursor
      node {
        id
        name
        tagline
        description
        votesCount
        commentsCount
        createdAt
        topics(first: 20) {
          edges {
            node {
              slug
              name
            }
          }
        }
        makers {
          id
        }
        media {
          url
          type
          videoUrl
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""


@dataclass
class LaunchRecord:
    post_id: str
    name: str
    tagline: str
    description: str
    votes_count: int
    comments_count: int
    created_at: str
    topics: str
    topic_count: int
    maker_count: int
    media_count: int


@dataclass
class WindowState:
    posted_after: str
    posted_before: str
    cap: int
    collected: int = 0
    cursor: str | None = None
    has_next_page: bool = True


def _request_posts(
    endpoint: str,
    token: str,
    first: int,
    posted_after: str,
    posted_before: str,
    after: str | None,
) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "query": POSTS_QUERY,
        "variables": {
            "first": first,
            "after": after,
            "postedAfter": posted_after,
            "postedBefore": posted_before,
        },
    }

    retries = 12
    for attempt in range(1, retries + 1):
        response: Response = requests.post(endpoint, headers=headers, json=payload, timeout=60)
        time.sleep(0.2)
        if response.status_code == 200:
            data = response.json()
            if "errors" in data:
                raise RuntimeError(f"GraphQL error: {data['errors']}")
            return data["data"]["posts"]

        if response.status_code == 429:
            try:
                response_data = response.json()
            except ValueError:
                response_data = {}

            details = (response_data.get("errors") or [{}])[0].get("details", {})
            reset_in = details.get("reset_in")
            if reset_in is None:
                retry_after = response.headers.get("Retry-After")
                reset_in = float(retry_after) if retry_after else 60.0
            sleep_seconds = max(float(reset_in), 1.0) + 2.0
            print(f"Rate limit reached. Waiting {int(sleep_seconds)} seconds before retrying...")
            time.sleep(sleep_seconds)
            continue

        if response.status_code == 403 and "Just a moment" in response.text:
            sleep_seconds = min(120 * attempt, 900)
            print(
                f"Temporary anti-bot challenge encountered (403). "
                f"Waiting {int(sleep_seconds)} seconds before retrying..."
            )
            time.sleep(sleep_seconds)
            continue

        retry_after = response.headers.get("Retry-After")
        if retry_after:
            sleep_seconds = float(retry_after)
        else:
            sleep_seconds = min(2**attempt, 30)

        if attempt == retries:
            raise RuntimeError(
                f"Product Hunt API request failed with status {response.status_code}: {response.text}"
            )
        time.sleep(sleep_seconds)

    raise RuntimeError("Unreachable retry loop.")


def _flatten_node(node: dict[str, Any]) -> LaunchRecord:
    topic_edges = node.get("topics", {}).get("edges", [])
    topic_names = [edge["node"]["name"] for edge in topic_edges if edge.get("node")]
    makers = node.get("makers") or []
    media = node.get("media") or []

    return LaunchRecord(
        post_id=node["id"],
        name=node.get("name") or "",
        tagline=node.get("tagline") or "",
        description=node.get("description") or "",
        votes_count=int(node.get("votesCount") or 0),
        comments_count=int(node.get("commentsCount") or 0),
        created_at=node.get("createdAt") or "",
        topics="|".join(topic_names),
        topic_count=len(topic_names),
        maker_count=len(makers),
        media_count=len(media),
    )


def _parse_iso_utc(raw_value: str) -> datetime:
    return datetime.fromisoformat(raw_value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _format_iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _build_windows(posted_after: str, posted_before: str, target: int, months_per_window: int) -> list[WindowState]:
    if months_per_window <= 0:
        raise ValueError("months_per_window must be > 0")

    start = _parse_iso_utc(posted_after)
    end = _parse_iso_utc(posted_before)
    if start >= end:
        raise ValueError("POSTED_AFTER must be earlier than POSTED_BEFORE")

    windows: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        window_end = min(cursor + timedelta(days=30 * months_per_window), end)
        windows.append((cursor, window_end))
        cursor = window_end

    cap_base = target // len(windows)
    remainder = target % len(windows)

    window_states: list[WindowState] = []
    for idx, (window_start, window_end) in enumerate(windows):
        cap = cap_base + (1 if idx < remainder else 0)
        window_states.append(
            WindowState(
                posted_after=_format_iso_utc(window_start),
                posted_before=_format_iso_utc(window_end),
                cap=cap,
            )
        )
    return window_states


def collect_posts(max_posts: int | None = None) -> pd.DataFrame:
    settings = load_settings()
    ensure_directories(settings)

    target = max_posts if max_posts is not None else settings.max_posts
    records: list[LaunchRecord] = []
    seen_post_ids: set[str] = set()
    windows = _build_windows(
        posted_after=settings.posted_after,
        posted_before=settings.posted_before,
        target=target,
        months_per_window=3,
    )

    progress = tqdm(total=target, desc="Collecting posts", unit="post")
    while len(records) < target:
        any_progress = False
        for window in windows:
            if len(records) >= target:
                break
            if not window.has_next_page or window.collected >= window.cap:
                continue

            posts = _request_posts(
                endpoint=settings.graphql_endpoint,
                token=settings.product_hunt_token,
                first=settings.page_size,
                posted_after=window.posted_after,
                posted_before=window.posted_before,
                after=window.cursor,
            )

            edges = posts["edges"]
            if not edges:
                window.has_next_page = False
                continue

            for edge in edges:
                node = edge["node"]
                post_id = node["id"]
                if post_id in seen_post_ids:
                    continue
                seen_post_ids.add(post_id)
                records.append(_flatten_node(node))
                window.collected += 1
                progress.update(1)
                any_progress = True

                if len(records) % 200 == 0:
                    checkpoint = pd.DataFrame([asdict(record) for record in records])
                    checkpoint.to_csv(settings.raw_data_path, index=False)

                if len(records) >= target or window.collected >= window.cap:
                    break

            page_info = posts["pageInfo"]
            window.has_next_page = bool(page_info["hasNextPage"])
            window.cursor = page_info["endCursor"]

        if not any_progress:
            break

    progress.close()
    frame = pd.DataFrame([asdict(record) for record in records])
    frame.to_csv(settings.raw_data_path, index=False)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Product Hunt launch data.")
    parser.add_argument("--max-posts", type=int, default=None, help="Override max posts limit.")
    args = parser.parse_args()

    frame = collect_posts(max_posts=args.max_posts)
    print(f"Saved {len(frame)} rows to data/raw_posts.csv")


if __name__ == "__main__":
    main()
