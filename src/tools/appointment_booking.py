import streamlit as st
from typing import Dict
from langchain.tools import BaseTool
from pydantic import ValidationError

from ..models.user_info import UserInfo
from ..utils.date_extractor import DateExtractor

import textwrap


class AppointmentBookingTool(BaseTool):
    """Tool for booking appointments with conversational form"""
    name = "appointment_booking"
    description = "Book appointments by collecting user information through conversation"

    def _run(self, query: str) -> str:
        """Execute the appointment booking process"""
        try:
            # Initialize the session state for form data if it does not exist
            st.session_state.booking_state = {
                'step': 'name',
                'data': {},
                'active': True
            }

            state = st.session_state.booking_state
            
            if not state['active']:
                return "Appointment booking is not currently active. Please say 'book appointment' to start."
            
            extracted_date = DateExtractor.extract_date(query)
            if extracted_date and 'date' not in state['data']:
                state['data']['date'] = extracted_date

            return self._handle_booking_step(state, query)
        
        except Exception as e:
            return f"Sorry there was an error processing your appointment request: {str(e)}"
        

    def _handle_booking_step(self, state: Dict, query: str) -> str:
        if state['step'] == 'name':
            if 'name' not in state['data']:
                return "I'd be happy to help you book an appointment! What's your full name?"
            else:
                state['step'] = 'phone'
                return f"Thank you, {state['data']['name']}! What's your phone number?"
            
        elif state['step'] == 'phone':
            if 'phone' not in state['data']:
                return "Please provide your phone number so we can contact you."
            else:
                state['step'] = 'email'
                return "Great! Now I need your email address."
        
        elif state['step'] == 'email':
            if 'email' not in state['data']:
                return "Please provide your email address for appointment confirmation."
            else:
                state['step'] = 'date'
                if 'date' not in state['data']:
                    return "When would you like to schedule your appointment? (e.g., 'next Monday', 'tomorrow', or a specific date)"
                else:
                    return self._complete_booking(state['data'])
            
        elif state['step'] == 'date':
            if 'date' not in state['data']:
                return "Please specify your preferred appointment date."
            else:
                return self._complete_booking(state['data'])
            
        return "Let me help you with booking an appointment. What's your name?"
    

    def _complete_booking(self, data: Dict) -> str:
        """Complete the booking process and reset state"""
        try:
            # Validate all collected data
            user_info = UserInfo(**data)

            # Save appointment
            appointment_summary = textwrap.dedent(f"""
            ✅ Appointment Booked Successfully!

            📋 Details:
            - Name: {user_info.name}
            - Phone: {user_info.phone}
            - Email: {user_info.email}
            - Date: {data['date']}

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