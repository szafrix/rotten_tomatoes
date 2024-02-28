import os
import argparse
import datetime as dt
import random
import pandas as pd

from requests import Session
from requests.models import Response

from time import sleep
from bs4 import BeautifulSoup
from typing import List, Tuple, Optional

from rt_scraper.logger import get_scraper_logger
from rt_scraper.scraper_utils import get_list_of_headers

logger = get_scraper_logger()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scrapes review quotes and review sentiment from Rotten Tomatoes"
    )
    parser.add_argument(
        "--n_movies_to_get",
        type=int,
        help="How many movies to iterate over during retrieval of quotes and scores.",
    )
    args = parser.parse_args()
    return args


class RottenTomatoesScraper:
    headers = get_list_of_headers()
    base_url = "https://www.rottentomatoes.com"
    consecutive_errors = 0
    max_allowed_consecutive_errors = 5
    results_folder = "rt_scraper/results/"

    def scrape_quotes_and_scores_in_bulk_from_popular_movies_pages(
        self, n_movies_to_get: int
    ):
        with Session() as session:
            self._make_result_folder_if_necessary()
            results_filepath = (
                self.results_folder + str(dt.datetime.now()).split(".")[0] + ".json"
            )
            if movie_urls := self.get_list_of_movie_urls(session, n_movies_to_get):
                scraped_data = []
                try:
                    for movie_url in movie_urls:
                        if data_for_movie := self.get_texts_and_scores_from_url(
                            session, movie_url
                        ):
                            scraped_data += data_for_movie
                except Exception as exc:
                    if len(scraped_data) == 0:
                        logger.critical(
                            f"Scraping failed - no data obtained, reason: {exc}, terminating"
                        )
                    else:
                        logger.critical(
                            f"Scraper failed - saving {len(scraped_data)} scraped movies to {results_filepath}."
                        )
                        df = self._make_result_dataframe(scraped_data)
                        df.to_json(results_filepath)
                else:
                    logger.info(
                        f"Scraping terminated successfully, saving results to {results_filepath}"
                    )
                    df = self._make_result_dataframe(scraped_data)
                    df.to_json(results_filepath)
            else:
                logger.critical("Scraper failed during obtaining of movie URLs.")

    def get_list_of_movie_urls(
        self, session: Session, n_movies_to_get: int
    ) -> List[Optional[str]]:
        url = self.base_url + "/napi/browse/movies_at_home/sort:popular"
        all_movie_links = set()
        after = None
        response_format_error_count = 0
        while (len(all_movie_links) < n_movies_to_get) or (
            response_format_error_count < 5
        ):
            params = {"after": after}
            if response := self.scrape_page(session, url, params):
                try:
                    resp_json = response.json()
                except:
                    response_format_error_count += 1
                    logger.warning(
                        f"Response format error for {url} {params}, trial {response_format_error_count} / 5"
                    )
                    continue
                else:
                    response_format_error_count = 0
                movie_links = set(
                    [
                        self.base_url + movie["mediaUrl"]
                        for movie in resp_json["grid"]["list"]
                    ]
                )

                if movie_links.issubset(all_movie_links):
                    logger.warning(
                        f"No new movies to be added, returning list of {len(all_movie_links)} movie URLs."
                    )
                    break

                all_movie_links.update(movie_links)

                if resp_json["pageInfo"]["hasNextPage"]:
                    after = resp_json["pageInfo"]["endCursor"]
                else:
                    logger.warning(
                        f"Reached maximum number of pages, terminating with list of {len(all_movie_links)} movie URLs."
                    )
                    break
        logger.info(f"Successfully obtained {len(all_movie_links)} URLs to movies.")
        return all_movie_links

    def scrape_page(self, session: Session, url: str, params={}) -> Response | None:
        try:
            session.headers.update(random.choice(self.headers))
            response = session.get(url, params=params, timeout=15)
            sleep(random.uniform(a=0.5, b=1.5))
            response.raise_for_status()
        except Exception as exc:
            logger.error(f"{exc} occurred during scraping of {url}")
            self.consecutive_errors += 1
            if self.consecutive_errors == self.max_allowed_consecutive_errors:
                logger.critical(
                    "Reached max number of consecutive errors, terminating the scraper."
                )
                raise RuntimeError("Reached max number of consecutive errors")
        else:
            logger.info(f"Successfully scraped {url} {params}")
            self.consecutive_errors = 0
            return response

    def get_texts_and_scores_from_url(
        self, session: Session, url: str
    ) -> List[Tuple[str]] | None:
        if response := self.scrape_page(session, url):
            if soup := self.response_to_soup(response):
                if baloons := soup.find_all("review-speech-balloon-deprecated"):
                    quotes = [baloon["reviewquote"] for baloon in baloons]
                    scores = [baloon["scorestate"] for baloon in baloons]
                    if len(quotes) != len(scores):
                        logger.error(
                            f"Length of quotes ({len(quotes)}) does not match length of scores ({len(scores)}) for {url}, skipping this website"
                        )
                        return None
                    else:
                        logger.info(f"Obtained {len(quotes)} chunks from {url}")
                        return [
                            (url, q, s) for q, s in zip(quotes, scores) if all([q, s])
                        ]
                else:
                    logger.warning(f"No review baloons for {url}")

    @staticmethod
    def response_to_soup(response: Response) -> BeautifulSoup | None:
        try:
            return BeautifulSoup(response.content, features="lxml")
        except Exception as exc:
            logger.error(f"Could not convert response to soup, reason: {exc}")

    @classmethod
    def _make_result_folder_if_necessary(self) -> None:
        if not os.path.exists(self.results_folder):
            os.mkdir(self.results_folder)

    @classmethod
    @staticmethod
    def _make_result_dataframe(raw_results: List[Tuple[str]]) -> pd.DataFrame:
        return pd.DataFrame(raw_results, columns=["url", "quote", "score"])


if __name__ == "__main__":
    args = parse_args()
    r = RottenTomatoesScraper()
    r.scrape_quotes_and_scores_in_bulk_from_popular_movies_pages(args.n_movies_to_get)
