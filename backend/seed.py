"""Seed script to create admin user via Supabase."""
import os
import secrets
import string
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()


def _generate_password(length: int = 16) -> str:
    """Generate a cryptographically secure random password."""
    alphabet = string.ascii_letters + string.digits + "!@#$%&*"
    return ''.join(secrets.choice(alphabet) for _ in range(length))


def seed_admin():
    """Create default admin user in Supabase.

    Reads ADMIN_EMAIL from env (defaults to admin@mediaguardx.com).
    Generates a strong random password — never hardcoded.
    """
    url = os.getenv("SUPABASE_URL")
    key = os.getenv("SUPABASE_SERVICE_KEY")

    if not url or not key:
        print("Error: SUPABASE_URL and SUPABASE_SERVICE_KEY must be set in .env")
        return

    supabase = create_client(url, key)

    email = os.getenv("ADMIN_EMAIL", "admin@mediaguardx.com")
    password = os.getenv("ADMIN_PASSWORD") or _generate_password()
    name = os.getenv("ADMIN_NAME", "Admin")

    try:
        # Create auth user
        result = supabase.auth.admin.create_user({
            "email": email,
            "password": password,
            "email_confirm": True,
            "user_metadata": {"name": name},
        })

        user_id = result.user.id
        print(f"Created auth user: {email} (ID: {user_id})")

        # Update profile to admin role
        supabase.table("profiles").update({"role": "admin"}).eq("id", user_id).execute()
        print("Updated profile role to admin")

        # Print credentials securely — only to stdout, never to logs
        print(f"\nAdmin credentials (save these — they will not be shown again):")
        print(f"  Email:    {email}")
        print(f"  Password: {password}")
        print(f"\nPlease change the password after first login!")

    except Exception as e:
        if "already been registered" in str(e).lower() or "already exists" in str(e).lower():
            print(f"User {email} already exists. Updating role to admin...")
            # Find and update existing user
            profiles = supabase.table("profiles").select("id").eq("email", email).execute()
            if profiles.data:
                supabase.table("profiles").update({"role": "admin"}).eq("id", profiles.data[0]["id"]).execute()
                print("Role updated to admin.")
        else:
            print(f"Error: {e}")


if __name__ == "__main__":
    seed_admin()
