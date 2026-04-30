# main.py
# CLI entry point for PolicyPro.
# Launches a minimal terminal interface for testing queries without the GUI.
# Also accepts --reset-password to wipe a user's credentials from the terminal
# when they are locked out and cannot recover via the GUI.
#
# Functions: main, run_cli, reset_password

import argparse
import sys
import json
import os
import getpass


def reset_password(username: str) -> None:
    # Wipes the password hash and security answer for a user account so
    # they can set a new password via the GUI login screen.

    from config import USERS_FILE

    if not os.path.exists(USERS_FILE):
        print(f"[ERROR] No users file found at {USERS_FILE}.")
        sys.exit(1)

    with open(USERS_FILE, "r", encoding="utf-8") as f:
        users = json.load(f)

    user = next((u for u in users if u["username"].lower() == username.lower()), None)
    if not user:
        print(f"[ERROR] No user found with username '{username}'.")
        sys.exit(1)

    user["password_hash"] = None
    user["security_answer_hash"] = None

    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, indent=2)

    print(f"[OK] Password reset for '{username}'. They can now set a new password via the login screen.")


def run_cli() -> None:
    # Minimal terminal interface that prompts for login credentials and
    # accepts HR policy queries in a loop until the user types quit.

    from src.agents.orchestrator import run_orchestrator
    from gui import authenticate_user

    print("HR Policy Assistant — CLI Mode")
    print("Type 'quit' to exit.\n")

    username = input("Username: ").strip()
    password = getpass.getpass("Password: ")

    user = authenticate_user(username, password)
    if not user:
        print("[ERROR] Invalid credentials.")
        sys.exit(1)

    user_id = user["user_id"]
    print(f"\nWelcome, {username}. Type your HR policy questions below.\n")

    import uuid
    session_id = str(uuid.uuid4())

    while True:
        try:
            query = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break

        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        if not query:
            continue

        result = run_orchestrator(query, user_id=user_id, session_id=session_id)

        print(f"\nAssistant: {result['answer']}\n")

        if result.get("new_profile_facts"):
            for fact in result["new_profile_facts"]:
                print(f"[Profile updated] {fact}")
            print()


def main():
    # Parses command line arguments and routes to either reset_password
    # or run_cli depending on the flags provided.

    parser = argparse.ArgumentParser(description="HR Policy Assistant")
    parser.add_argument(
        "--reset-password",
        metavar="USERNAME",
        help="Reset a user's password (terminal access required)"
    )

    args = parser.parse_args()

    if args.reset_password:
        reset_password(args.reset_password)
    else:
        run_cli()


if __name__ == "__main__":
    main()