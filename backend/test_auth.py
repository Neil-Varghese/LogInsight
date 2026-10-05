import tempfile
import unittest
from pathlib import Path

import auth
from auth import Auth, AuthError


class AuthTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.auth = Auth(Path(self.dir.name) / "auth.sqlite3")

    def tearDown(self):
        self.dir.cleanup()

    def test_signup_login_logout(self):
        self.auth.signup("Dev@Example.com", "correct horse")
        token = self.auth.login("dev@example.com", "correct horse", "1.1.1.1")
        self.assertEqual(self.auth.user(token), {"email": "dev@example.com"})
        self.auth.logout(token)
        self.assertIsNone(self.auth.user(token))

    def test_password_is_not_stored_in_plain_text(self):
        self.auth.signup("a@b.co", "correct horse")
        with self.auth._connect() as db:
            stored = db.execute("SELECT salt, hash FROM users").fetchone()
        self.assertNotIn(b"correct horse", stored[0] + stored[1])

    def test_bad_input_and_duplicates(self):
        for email, password, status in [("nope", "correct horse", 400), ("a@b.co", "short", 400)]:
            with self.assertRaises(AuthError) as ctx:
                self.auth.signup(email, password)
            self.assertEqual(ctx.exception.status, status)
        self.auth.signup("a@b.co", "correct horse")
        with self.assertRaises(AuthError) as ctx:
            self.auth.signup("A@B.co", "another pass")
        self.assertEqual(ctx.exception.status, 409)

    def test_wrong_password_then_lockout(self):
        self.auth.signup("a@b.co", "correct horse")
        for _ in range(auth.MAX_FAILS):
            with self.assertRaises(AuthError) as ctx:
                self.auth.login("a@b.co", "wrong password", "9.9.9.9")
            self.assertEqual(ctx.exception.status, 401)
        with self.assertRaises(AuthError) as ctx:  # now locked, even with the right password
            self.auth.login("a@b.co", "correct horse", "9.9.9.9")
        self.assertEqual(ctx.exception.status, 429)
        self.assertTrue(self.auth.login("a@b.co", "correct horse", "8.8.8.8"))  # another address is not locked

    def test_expired_session_is_rejected(self):
        self.auth.signup("a@b.co", "correct horse")
        token = self.auth.login("a@b.co", "correct horse", "1.1.1.1")
        with self.auth._connect() as db:
            db.execute("UPDATE sessions SET expires = 0")
        self.assertIsNone(self.auth.user(token))


if __name__ == "__main__":
    unittest.main()
