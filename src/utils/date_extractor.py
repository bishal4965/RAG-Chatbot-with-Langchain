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
            r'next mon(?:day)?': DateExtractor._get_next_weekday(today, 0),
            # Use r'next mon(?:day)?' for matching 'next mon' as well as 'next monday' pattern
            r'next tue(?:sday)?': DateExtractor._get_next_weekday(today, 1),
            r'next wed(?:nesday)?': DateExtractor._get_next_weekday(today, 2),
            r'next thu(?:rsday)?': DateExtractor._get_next_weekday(today, 3),
            r'next fri(?:day)?': DateExtractor._get_next_weekday(today, 4),
            r'next sat(?:urday)?': DateExtractor._get_next_weekday(today, 5),
            r'next sun(?:day)?': DateExtractor._get_next_weekday(today, 6),
        }

        for pattern, date_obj in date_patterns.items():
            if re.search(pattern, text):
                return date_obj.strftime('%Y-%m-%d')
            
        # Handle specific date formats
        date_regex_patterns = [
            r'(\d{1,2})/(\d{1,2})/(\d{4})',     # MM/DD/YYYY
            r'(\d{4})-(\d{1,2})-(\d{1,2})',     # YYYY-MM-DD
            r'(\d{1,2})-(\d{1,2})-(\d{4})',     # DD-MM-YYYY
        ]

        for pattern in date_regex_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    if pattern == r'(\d{4})-(\d{1,2})-(\d{1,2})':
                        year, month, day = match.groups()
                    elif pattern == r'(\d{1,2})/(\d{1,2})/(\d{4})':
                        month, day, year = match.groups()
                    else:  # DD-MM-YYYY
                        day, month, year = match.groups()
                    
                    date_obj = datetime(int(year), int(month), int(day))
                    return date_obj.strftime('%Y-%m-%d')
                except ValueError:
                    continue

        return None
    
    
    @staticmethod
    def _get_next_weekday(date: datetime, weekday: int) -> datetime:
        """Get the next occurrence of a specific weekday"""
        days_ahead = weekday - date.weekday()
        if days_ahead <= 0:
            days_ahead += 7
        return date + timedelta(days_ahead)