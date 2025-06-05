import streamlit as st
from typing import Dict
from langchain.tools import BaseTool
from pydantic import ValidationError

from ..models.user_info import UserInfo
from ..utils.date_extractor import DateExtractor
from config.logging import logger

import textwrap


class AppointmentBookingTool(BaseTool):
    """Tool for booking appointments with conversational form"""
    name: str = "appointment_booking"
    description: str = "Book appointments by collecting user information through conversation"

    def _run(self, query: str) -> str:
        """Execute the appointment booking process"""

        # logger.debug("→ booking_state at start of _run:", st.session_state.get('booking_state'))
        try:
            # Initialize the session state for form data if it does not exist
            if 'booking_state' not in st.session_state:
                st.session_state.booking_state = {
                    'step': 'name',
                    'data': {},
                    'active': False
                }

            state = st.session_state.booking_state

            # Check if this is a booking initiation request
            booking_keywords = ['book appointment', 'schedule', 'call me', 'contact me', 'appointment']
            is_booking_request = any(keyword in query.lower() for keyword in booking_keywords)
            
            if is_booking_request and not state['active']:
                state['active'] = True
                state['step'] = 'name'
                state['data'] = {}

                return "I'd be happy to help you book an appointment! What's your full name?"
            
            if not state['active']:
                return "Appointment booking is not currently active. Please say 'book appointment' to start."

            return self._handle_booking_input(query)
        
        except Exception as e:
            return f"Sorry there was an error processing your appointment request: {str(e)}"
      
    def _validate_field(self, field_name: str, value: str) -> tuple[bool, str, str]:
        """Validate individual field using UserInfo Model"""

        try:
            # Minimal UserInfo obj for placeholder
            test_data = {
                'name': 'Test User',
                'phone': '+442083661177',
                'email': 'test@example.com'
            }
            test_data[field_name] = value

            user_info = UserInfo(**test_data)
            # Dynamically access attribute of an object
            cleaned_value = getattr(user_info, field_name)

            return True, cleaned_value, ""

        except ValidationError as e:
            for error in e.errors():
                logger.debug(f"→ ValidationError: {error}")
                # Return the first validation error regardless of field
                return False, value, error['msg']
            return False, value, "Validation failed"


    def _handle_field(self, field_name: str, user_input: str, next_step: str, prompt: str) -> str:
            """Helper function to handle field and clean input"""

            state = st.session_state.booking_state

            # DEBUG: See what we are going to validate
            logger.debug(f"→ _handle_field called with field_name={field_name!r}, user_input={user_input!r}, step={state['step']!r}")


            is_valid, cleaned_value, error_msg = self._validate_field(field_name, user_input)
            logger.debug(f"-> is_valid value: {is_valid}, cleaned_value: {cleaned_value}, error_msg: {error_msg}")

            if is_valid:
                
                state['data'][field_name] = cleaned_value
                state['step'] = next_step
                logger.debug(f"→ setting data[{field_name!r}]={cleaned_value!r}, advancing step to {next_step!r}")
                return f"Thank you, {cleaned_value}! {prompt}" if field_name == 'name' else prompt
            
            return f"Please provide a valid {field_name}. {error_msg}" 


    def _handle_booking_input(self, user_input: str) -> str:
        """Handle input during booking process"""
        
        state = st.session_state.booking_state

        try:
            if 'date' not in state['data']:
                extracted_date = DateExtractor.extract_date(user_input)
                if extracted_date:
                    state['data']['date'] = extracted_date

            if state['step'] == 'name' and 'name' not in state['data']:
                return self._handle_field('name', user_input, next_step='phone', prompt="What's your phone number?")
                    
            elif state['step'] == 'phone' and 'phone' not in state['data']:
               return self._handle_field('phone', user_input, next_step='email', prompt="Great! Now I need your email address.")                  
            
            elif state['step'] == 'email' and 'email' not in state['data']:
                is_valid, cleaned_email, error_msg = self._validate_field('email', user_input)
                if is_valid:
                    state['data']['email'] = cleaned_email
                    state['step'] = 'date'

                    # If we already have a date, complete booking
                    if 'date' in state['data']:
                        return self._complete_booking()
                    else:
                        return "When would you like to schedule your appointment? (e.g., 'next Monday', 'tomorrow', or a specific date like '2025-06-02')"
                else:
                    return f"Please provide a valid email address. {error_msg}"
                
            elif state['step'] == 'date':
                if 'date' not in state['data']:
                    extracted_date = DateExtractor.extract_date(user_input)
                    if extracted_date:
                        state['data']['date'] = extracted_date
                        return self._complete_booking()
                    else:
                        return "I couldn't understand the date. Please try again with formats like 'next Monday', 'tomorrow', or 'YYYY-MM-DD'."
                else:
                    return self._complete_booking()
                
            else:
                return self._complete_booking()
            
        except Exception as e:
            return f"Sorry, there was an error: {str(e)}"
    

    def _complete_booking(self) -> str:
        """Complete the booking process and reset state"""
        
        state = st.session_state.booking_state

        try:
            # Validate all collected data
            user_info = UserInfo(**state['data'])

            # Save appointment
            appointment_summary = textwrap.dedent(f"""
            ✅ Appointment Booked Successfully!

            📋 Details:
            - Name: {user_info.name}
            - Phone: {user_info.phone}
            - Email: {user_info.email}
            - Date: {state['data']['date']}

            You'll receive a confirmation email shortly. Is there anything else I can help you with?
            """)
            
            # Reset booking state
            st.session_state.booking_state = {
                'step': 'name',
                'data': {},
                'active': False
            }

            return appointment_summary.strip()
        
        except ValidationError as e:
            error_messages = [] 
            for error in e.errors():
                field = error['loc'][0]
                message = error['message']
                error_messages.append(f"{field}: {message}")
                
            return "Please correct the following information:\n" + "\n".join(error_messages)
        
    
    def is_booking_active(self) -> bool:
        """Check if booking process is currently active"""
        return (hasattr(st.session_state, 'booking_state') and st.session_state.booking_state.get('active', False))
    
    
    def reset_booking(self) -> None:
        """Reset the booking state"""
        if hasattr(st.session_state, 'booking_state'):
            st.session_state.booking_state = {
                'step': 'name',
                'data': {},
                'active': False
            }

    def booking_progress(self) -> Dict:
        """Get current booking progress for monitoring"""
        if hasattr(st.session_state, 'booking_state'):
            return {
                'active': st.session_state.booking_state.get('active', False),
                'step': st.session_state.booking_state.get('step', 'name'),
                'completed_fields': list(st.session_state.booking_state.get('data', {}).keys())
            }
        
        return {'active': False, 'step': 'name','completed_fields': []}