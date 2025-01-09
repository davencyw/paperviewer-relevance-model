#!/usr/bin/env python

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

import curses
import textwrap
from typing import List

from tqdm import tqdm

from dataset import Paper


class TextAnnotationTUI:
    """
    Simple TUI to allow the user to annotate a list of papers given their titles and abstracts.
    """

    def __init__(self, papers: List[Paper]):
        """
        Initialize the TextAnnotationTUI.

        Args:
            papers: The list of papers to annotate.
        """
        self.papers = papers

    def display(self, stdscr: curses.window) -> None:
        """
        Main function to display the TUI to the user.

        Args:
            stdscr: The standard curses window.
        """
        for paper_i in tqdm(range(len(self.papers)), desc="Annotating papers"):
            title = self.papers[paper_i].title
            abstract = self.papers[paper_i].abstract

            curses.curs_set(0)  # Disable cursor
            stdscr.clear()

            # Get terminal dimensions
            height, width = stdscr.getmaxyx()

            # Center the title vertically and horizontally
            title_lines = textwrap.wrap(title, width - 2)
            title_height = len(title_lines)
            title_start_y = 3
            for i, line in enumerate(title_lines):
                x_position = max((width - len(line)) // 2, 0)  # Center the line horizontally
                stdscr.addstr(title_start_y + i, x_position, line, curses.A_BOLD)

            # Display the abstract below the title
            abstract_lines = textwrap.wrap(abstract, width - 2)
            abstract_start_y = title_start_y + title_height + 2
            stdscr.addstr(abstract_start_y, 0, "Abstract:", curses.A_BOLD)
            for i, line in enumerate(abstract_lines, start=1):
                stdscr.addstr(abstract_start_y + i, 0, line)

            # Display instructions
            instruction_line = abstract_start_y + len(abstract_lines) + 2
            stdscr.addstr(
                instruction_line,
                0,
                "Instructions: Press 'x' for miss, or a value between 0-10 for valuable.",
                curses.A_DIM,
            )

            # Wait for user input
            while True:
                try:
                    stdscr.addstr(instruction_line + 2, 0, "Enter your label: ")
                except curses.error:
                    pass
                stdscr.refresh()
                user_input = stdscr.getkey()
                try:
                    stdscr.addstr(
                        instruction_line,
                        0,
                        f"Instructions: Press 'x' for miss, or a value between 0-10 for valuable. Pressed: {user_input}",
                        curses.A_DIM,
                    )
                except curses.error:
                    pass

                # Handle input
                if user_input.lower() == "x":
                    current_annotation = -1
                    break
                if user_input == "-":
                    current_annotation = 10
                elif user_input.isdigit() and 0 <= int(user_input) <= 10:
                    current_annotation = int(user_input)
                    break
                else:
                    try:
                        stdscr.addstr(
                            instruction_line + 3,
                            0,
                            "Invalid input. Please press 'x' or a value between 0-10.",
                            curses.A_DIM,
                        )
                    except curses.error:
                        pass

            self.papers[paper_i].annotation = current_annotation
            stdscr.clear()

    def run(self) -> None:
        """
        Runs the review UI.

        This function will start the curses UI and begin displaying the papers.
        """
        # Use curses.wrapper to ensure that the terminal is returned to its
        # original state after the program exits.
        curses.wrapper(self.display)

    def report(self, verbose: bool = True) -> float:
        """
        Prints a report to the console of the average annotation, and how many
        papers were missed.

        Args:
            verbose: Whether to print the report to the console.

        Returns:
            The percentage of missed papers.
        """
        # Count the number of papers that were missed and annotated
        rated_papers = self.papers
        num_missed = len([paper for paper in rated_papers if paper.annotation == -1])
        num_annotated = len([paper for paper in rated_papers if paper.annotation != -2])

        # Calculate the average annotation
        average_annotation = sum([paper.annotation for paper in rated_papers]) / len(rated_papers)

        # Calculate the percentage of missed and annotated papers
        percentage_missed = num_missed / len(rated_papers)
        percentage_annotated = num_annotated / len(rated_papers)

        # Print the report
        if verbose:
            tqdm.write(
                f"Average annotation: {average_annotation},\n"
                f"Number of missed: {num_missed}, Percentage missed: {percentage_missed*100:.2f}%, \n"
                f"Number of annotated: {num_annotated}, Percentage annotated: {percentage_annotated*100:.2f}%"
            )

        # Return the percentage of missed papers
        return percentage_missed


if __name__ == "__main__":
    # Example usage of the TUI, run with $ ./src/review_ui.py
    test_papers = Paper.list_from_json("test_data/paper_samples.json")
    tui = TextAnnotationTUI(test_papers)
    tui.run()
    tui.report()
