"""
Access Control Module
Manages 3-level access control (Public, Student, Faculty)
"""

from typing import Dict, List, Optional
from .config import Config


class AccessController:
    """Handles access level permissions and filtering"""
    
    def __init__(self):
        self.access_levels = Config.ACCESS_LEVELS
        self.hierarchy = Config.ACCESS_HIERARCHY
    
    def get_accessible_levels(self, user_level: str) -> List[str]:
        """
        Get all access levels a user can see
        
        Args:
            user_level: User's access level (public, student, faculty)
            
        Returns:
            List of accessible levels
        """
        return self.hierarchy.get(user_level, ["public"])
    
    def build_filter(
        self, 
        user_level: str, 
        user_id: Optional[str] = None
    ) -> Dict:
        """
        Build ChromaDB where filter for access control
        
        Args:
            user_level: User's access level
            user_id: Optional user ID for personal documents
            
        Returns:
            ChromaDB where clause dictionary
        """
        accessible_levels = self.get_accessible_levels(user_level)
        
        # Base filter - user can access these levels
        base_filter = {"access_level": {"$in": accessible_levels}}
        
        # If user has ID, also include their personal documents
        if user_id:
            return {
                "$or": [
                    {"access_level": {"$in": accessible_levels}},
                    {"owner": user_id}
                ]
            }
        
        return base_filter
    
    def get_level_description(self, level: str) -> str:
        """Get human-readable description of access level"""
        return self.access_levels.get(level, "Unknown")
    
    def get_permissions_text(self, user_level: str) -> str:
        """
        Get text description of what user can access
        
        Args:
            user_level: User's access level
            
        Returns:
            Formatted string of permissions
        """
        if user_level == "public":
            return "✓ Public documents"
        elif user_level == "student":
            return "✓ Public documents\n✓ Student documents"
        elif user_level == "faculty":
            return "✓ All documents"
        else:
            return "✓ Public documents (default)"