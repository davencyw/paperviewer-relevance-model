"""
This file is part of the Relevance Model System.
Copyright (C) 2025, David Schmidig, dave@davencyw.com

The Relevance Model System is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

The Relevance Model System is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with The Relevance Model System.  If not, see <https://www.gnu.org/licenses/>.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Any, List, Optional

DEFAULT_DATABASE = "database/database.db"


@dataclass
class Paper:
    title: str
    authors: List[str]
    conferences: Optional[str]
    comment: str
    abstract: str
    categories: List[str]
    id: str
    pdf_url: str
    pdf_file: Optional[str]
    published: datetime
    source: str
    score: Optional[float]
    selected_review: Optional[bool] = None
    selected_review_date: Optional[datetime] = None


def convert_to_paper_instance(paper_data: List[Any]) -> Paper:
    """
    Convert a list of paper data, as read from the database, into a Paper instance.

    Args:
        paper_data: A list of paper data, as read from the database.

    Returns:
        A Paper instance, created from the given data.
    """
    return Paper(
        title=paper_data[1],
        authors=paper_data[2].split(";"),
        conferences=paper_data[3],
        comment=paper_data[4],
        abstract=paper_data[5],
        categories=paper_data[6].split(";"),
        id=paper_data[0],
        pdf_url=paper_data[7],
        pdf_file=paper_data[8],
        published=datetime.strptime(paper_data[9].split(" ")[0], "%Y-%m-%d"),
        source=paper_data[10],
        score=paper_data[11],
        selected_review=paper_data[12],
        selected_review_date=paper_data[13],
    )


class PaperDatabaseManager:
    """
    Class to interact with the paper database and extract papers from it.
    """

    def __init__(self, db_name: str = DEFAULT_DATABASE):
        self.conn = sqlite3.connect(db_name)
        self.cursor = self.conn.cursor()

    def get_all_papers(self) -> List[Paper]:
        return self.retrieve_paper_since(datetime.utcfromtimestamp(0))

    def retrieve_paper_since(self, since: datetime) -> List[Paper]:
        return self.get_paper_between_dates(since, datetime.now())

    def get_paper_between_dates(
        self, inclusive_start_date: datetime, inclusive_end_date: datetime, selected: bool = False
    ) -> List[Paper]:
        if selected:
            self.cursor.execute(
                "SELECT * FROM papers WHERE selected_review_date >= ? AND selected_review_date <= ? AND selected_review = ?",
                (inclusive_start_date, inclusive_end_date, True),
            )
        else:
            self.cursor.execute(
                "SELECT * FROM papers WHERE published >= ? AND published <= ?",
                (inclusive_start_date, inclusive_end_date),
            )
        all_papers = self.cursor.fetchall()

        all_papers_converted = [convert_to_paper_instance(paper_data) for paper_data in all_papers]
        return all_papers_converted
