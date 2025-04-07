"""Test factories."""

from factory.faker import Faker

Faker._DEFAULT_LOCALE = "en_US"

from tests.factories.models import *
