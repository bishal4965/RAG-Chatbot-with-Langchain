import re
from pydantic import BaseModel, field_validator
import phonenumbers
from phonenumbers import NumberParseException

from config.logging import logger

class UserInfo(BaseModel):
    """Pydantic model for user information validation"""
    name: str
    phone: str
    email: str

    @field_validator('email')
    def validate_email(cls, v):
        email_pattern = r'^[a-zA-Z0-9+_.%-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        # ^ ensures email starts with valid chars and $ ensures nth comes after TLD
        # ^ anchors the pattern to start of the string
        # $ anchors the pattern at the end of the string
        if not re.match(email_pattern, v):
            raise ValueError('Invalid email format')
        return v
    
    @field_validator('phone')
    def validate_phone(cls, v):
        logger.debug("→ validate_phone got: %s", repr(v))
        try:
            parsed = phonenumbers.parse(v, None)
            if not phonenumbers.is_valid_number(parsed):
                raise ValueError('Invalid phone number')
            return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.INTERNATIONAL)
        except NumberParseException:
            raise ValueError('Invalid phone number format')
        
    @field_validator('name')
    def validate_name(cls, v):
        logger.debug("→ validate_name received: %s", repr(v))
        if len(v.strip()) < 2:
            raise ValueError('Name must be at least 2 characters long')
        return v.strip()

