import re
from datetime import datetime, timedelta
from typing import Optional


class DateExtractor:
    """Utility class to extract and parse dates from natural language"""

    @staticmethod
    def extract_date(text: str) -> Optional[str]:
        """Extract date from natural language and return YYYY-MM-DD format"""
        text = text.lower()
        today = datetime.now()

        # Handle specific date patterns
        date_patterns = {
            r'today': today,
            r'tomorrow': today + timedelta(days=1),
            r'next monday': DateExtractor._get_next_weekday(today, 0),
            # Use r'next mon(?:day)?' for matching 'next mon' as well as 'next monday' pattern
            r'next tuesday': DateExtractor._get_next_weekday(today, 1),
            r'next wednesday': DateExtractor._get_next_weekday(today, 2),
            r'next thursday': DateExtractor._get_next_weekday(today, 3),
            r'next friday': DateExtractor._get_next_weekday(today, 4),
            r'next saturday': DateExtractor._get_next_weekday(today, 5),
            r'next sunday': DateExtractor._get_next_weekday(today, 6),
        }

        for pattern, date_obj in date_patterns.items():
            if re.search(pattern, text):
                return date_obj.strftime('%Y-%m-%d')
            
        # # Handle specific date formats
        # date_regex_patterns = [
        #     r'(\d{1,2})/(\d{1,2})/(\d{4})',     # MM/DD/YYYY
        #     r'(\d{4})-(\d{1,2})-(\d{1,2})',     # YYYY-MM-DD
        #     r'(\d{1,2})-(\d{1,2})-(\d{4})',     # DD-MM-YYYY
        # ]

    @staticmethod
    def _get_next_weekday(date: datetime, weekday: int) -> datetime:
        """Get the next occurrence of a specific weekday"""
        days_ahead = weekday - date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return date + timedelta(days_ahead)