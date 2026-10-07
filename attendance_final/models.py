"""
models.py
---------
Demonstrates core OOP concepts required by the syllabus:
    - Abstraction   (Person is an abstract base class)
    - Inheritance   (Student inherits from Person)
    - Encapsulation (protected attributes with getters/setters)
    - Polymorphism  (display_info() is overridden differently in each subclass)
"""

from abc import ABC, abstractmethod


class Person(ABC):
    """Abstract base class. You cannot create a bare Person - every person
    in this system must be a more specific kind of person."""

    def __init__(self, name: str, roll_no: str):
        self._name = name
        self._roll_no = roll_no

    @property
    def name(self) -> str:
        return self._name

    @property
    def roll_no(self) -> str:
        return self._roll_no

    @abstractmethod
    def display_info(self) -> str:
        raise NotImplementedError

    def __str__(self) -> str:
        return self.display_info()


class Student(Person):
    """A Student IS-A Person, plus fields specific to students."""

    def __init__(self, name: str, roll_no: str, department: str, semester: int,
                 student_id: int = None, is_active: bool = True):
        super().__init__(name, roll_no)
        self._department = department
        self._semester = semester
        self.student_id = student_id
        self.is_active = is_active  # plain attribute: an administrative status
                                     # flag, not core identity - deliberately not
                                     # protected/property-wrapped like name/roll_no

    @property
    def department(self) -> str:
        return self._department

    @property
    def semester(self) -> int:
        return self._semester

    def display_info(self) -> str:
        status = "" if self.is_active else " [inactive]"
        return (f"{self.name} (Roll {self.roll_no}) - "
                f"{self.department}, Semester {self.semester}{status}")


class User:
    """A login account: username, password, and a role that decides what
    the account is allowed to do. Not every User is a Student (Admin and
    Teacher accounts aren't), so this is composition, not inheritance - a
    User can optionally be *linked to* a Student via student_id rather
    than *being* one."""

    VALID_ROLES = ("admin", "teacher", "student")

    def __init__(self, username: str, role: str, user_id: int = None,
                 student_id: int = None):
        if role not in User.VALID_ROLES:
            raise ValueError(f"Unknown role: {role}")
        self.username = username
        self.role = role
        self.user_id = user_id
        self.student_id = student_id  # only set for role == "student"
