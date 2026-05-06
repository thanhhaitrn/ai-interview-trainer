# SQLite / source of truth
#→ lưu fact dạng structured dict/json

#Embedding text
#→ câu tiếng Việt rõ nghĩa để semantic search tốt

#Vector metadata
#→ field ngắn để filter chính xác

import sqlite3
import json
from typing import Any

def insert_resume(conn: sqlite3.Connection, data: dict[str, Any]) -> int:
    """Insert one parsed resume document into the resumes table."""

    source = data.get("source", {})
    candidate = data.get("candidate", {})

    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO resumes_doc (
            file_name,
            page_count,
            name,
            headline,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            source.get("file_name"),
            source.get("page_count"),
            candidate.get("full_name"),
            candidate.get("headline"),
            json.dumps(data, ensure_ascii=False)
        ),
    )

     # Return id of this so other tables can link back to this resume.
    return int(cur.lastrowid)

def insert_work_experiences(
    conn: sqlite3.Connection,
    resume_id: int,
    item: dict[str, Any],
) -> None:
    """
    Insert one work experiences item into:
    1. work_experiences
    2. work_bullets
    """

    cur = conn.cursor()

    # This stores the job-level details:
    # - company
    # - job title
    # - location
    # - start date
    # - end date
    # - whether this is the candidate's current job

    cur.execute(
        """
        INSERT INTO work_experiences (
            resume_id,
            company,
            job_title,
            location,
            start_date,
            end_date,
            is_current
        )
        VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
        (
            # resume_id links this job back to the parent resume.
            resume_id,
            item.get("company"),
            item.get("job_title"),
            item.get("location"),
            item.get("start_date"),
            item.get("end_date"),
            # True  -> 1
            # False -> 0
            int(item.get("is_current", False)),
        ),
    )

    work_experience_id = int(cur.lastrowid)

    # Insert each bullet point into the work_bullets table.
    for bullet in item.get("bullets", []):
        cur.execute(
            """
            INSERT INTO work_bullets (
                work_experience_id,
                bullet_text
            )
            VALUES (?, ?);
            """,
            (work_experience_id, bullet),
        )


def insert_project(
    conn: sqlite3.Connection,
    resume_id: int,
    item: dict[str, Any],
) -> None:
    # Insert the project-level information first.
    # This only stores the project name.
    # The detailed bullet points go into project_bullets.
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO projects (
            resume_id,
            project_name
        )
        VALUES (?, ?);
        """,
        (
            resume_id,
            item.get("project_name"),
        ),
    )

    project_id = int(cur.lastrowid)

    for bullet in item.get("bullets", []):
        cur.execute(
            """
            INSERT INTO project_bullets (
                project_id,
                bullet_text
            )
            VALUES (?, ?);
            """,
            (project_id, bullet),
        )


def insert_skill(
    conn: sqlite3.Connection,
    resume_id: int,
    item: dict[str, Any],
) -> None:
    """
    Insert skills into the skills table. Each bullet becomes one row in the skills table.
    """
     
    cur = conn.cursor()

    for bullet in item.get("bullets", []):
        cur.execute(
            """
            INSERT INTO skills (
                resume_id,
                skill_text,
                category
            )
            VALUES (?, ?, ?);
            """,
            (
                resume_id,
                bullet,
                None,
            ),
        )


def insert_education(
    conn: sqlite3.Connection,
    resume_id: int,
    item: dict[str, Any],
) -> None:
    """
    Insert one education item into:
    1. education
    2. education_courses
    """

    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO education (
            resume_id,
            degree,
            field_of_study,
            institution,
            location,
            start_date,
            end_date,
            gpa
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?);
        """,
        (
            # This stores:
            # - degree
            # - field of study
            # - institution
            # - location
            # - dates
            # - GPA
            resume_id,
            item.get("degree"),
            item.get("field_of_study"),
            item.get("institution"),
            item.get("location"),
            item.get("start_date"),
            item.get("end_date"),
            item.get("gpa"),
        ),
    )

    education_id = int(cur.lastrowid)

    for course in item.get("relevant_courses", []):
        cur.execute(
            """
            INSERT INTO education_courses (
                education_id,
                course_name
            )
            VALUES (?, ?);
            """,
            (education_id, course),
        )

def insert_resume_json(conn: sqlite3.Connection, data: dict[str, Any]) -> int:
    """
    Insert the full parsed resume JSON into all relevant tables.
    Step 1:
        Insert the parent resume row into the resumes table.
    Step 2:
        Loop through each section in the JSON.
    Step 3:
        Depending on section_name, send each item to the correct insert function.
    Returns:
        resume_id of the inserted resume.
    """

    # Insert the parent resume document first.
    # This creates one row in the resumes table.
    # The returned resume_id will be used to connect all child records back to this resume.
    resume_id = insert_resume(conn, data)

    # Loop through all sections in the JSON.
    for section in data.get("sections", []):

        # Get the section name so we know which table to insert into.
        section_name = section.get("section_name")

        # Each section has a list of items.
        for item in section.get("items", []):

            # If this is a Work Experience section,insert into work_experiences and work_bullets.
            if section_name == "Work Experience":
                insert_work_experiences(conn, resume_id, item)

            # If this is a Projects section, insert into projects and project_bullets.
            elif section_name == "Projects":
                insert_project(conn, resume_id, item)

            # If this is a Skills section, insert skill rows.
            elif section_name == "Skills":
                insert_skill(conn, resume_id, item)

            # If this is an Education section, insert into education and education_courses.
            elif section_name == "Education":
                insert_education(conn, resume_id, item)

            # If the parser later creates an unknown section, this current version simply skips it. Later, you can add an "others" table if needed.
            else:
                pass

    # Return the parent resume id.
    return resume_id
