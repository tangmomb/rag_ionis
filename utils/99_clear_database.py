import os

import psycopg
from dotenv import load_dotenv


TABLES = (
    "comments",
    "video_transcripts",
    "video_daily_stats",
    "videos",
)


def main():
    load_dotenv()

    with psycopg.connect(os.environ["DATABASE_URL"]) as connection:
        with connection.cursor() as cursor:
            cursor.execute(f"TRUNCATE {', '.join(TABLES)} RESTART IDENTITY CASCADE")

    print("Base videe")


if __name__ == "__main__":
    main()
