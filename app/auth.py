"""Identify the user from Cloudflare Access.

The app listens only on 127.0.0.1 and is reached through the Cloudflare Tunnel,
but we still verify the signed Access JWT rather than trusting a plain header.
"""

from dataclasses import dataclass
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, Request
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.config import Settings, get_settings
from app.db import get_session
from app.models import User


@dataclass
class CurrentUser:
    id: int
    email: str
    is_admin: bool


@lru_cache
def _jwks_client(team_domain: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(f"https://{team_domain}/cdn-cgi/access/certs", cache_keys=True)


def email_from_request(request: Request, settings: Settings) -> str:
    if settings.dev_user_email:
        return settings.dev_user_email.lower()
    token = request.headers.get("cf-access-jwt-assertion")
    if not token or not settings.cf_access_team_domain or not settings.cf_access_aud:
        raise HTTPException(status_code=401, detail="Not authenticated")
    try:
        key = _jwks_client(settings.cf_access_team_domain).get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            key.key,
            algorithms=["RS256"],
            audience=settings.cf_access_aud,
            issuer=f"https://{settings.cf_access_team_domain}",
        )
    except jwt.PyJWTError as exc:
        raise HTTPException(status_code=401, detail="Invalid Access token") from exc
    email = claims.get("email")
    if not email:
        raise HTTPException(status_code=401, detail="No email in Access token")
    return email.lower()


def current_user(
    request: Request,
    session: Session = Depends(get_session),
    settings: Settings = Depends(get_settings),
) -> CurrentUser:
    email = email_from_request(request, settings)
    user = session.scalar(select(User).where(User.email == email))
    if user is None:
        try:
            user = User(email=email)
            session.add(user)
            session.commit()
        except IntegrityError:  # created concurrently by another request
            session.rollback()
            user = session.scalar(select(User).where(User.email == email))
    return CurrentUser(id=user.id, email=email, is_admin=email in settings.admins)
