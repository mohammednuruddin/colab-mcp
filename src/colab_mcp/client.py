import base64
from enum import Enum
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urljoin, urlparse
import json
import logging
import uuid

import requests
from pydantic import BaseModel, Field, TypeAdapter

# From src/colab/headers.ts
ACCEPT_JSON_HEADER = {"key": "Accept", "value": "application/json"}
AUTHORIZATION_HEADER = {"key": "Authorization", "value": ""}
COLAB_CLIENT_AGENT_HEADER = {
    "key": "X-Goog-Colab-Client-Agent",
    "value": "python-colab-client",
}
COLAB_XSRF_TOKEN_HEADER = {"key": "X-Goog-Colab-Token", "value": ""}


def uuid_to_web_safe_base64(notebook_hash: uuid.UUID) -> str:
    return base64.urlsafe_b64encode(notebook_hash.bytes).rstrip(b"=").decode("utf-8")


class Accelerator(str, Enum):
    NONE = "ACCELERATOR_NONE"
    T4 = "T4"
    L4 = "L4"
    A100 = "A100"
    V28 = "V2-8"
    V5E1 = "V5E-1"
    V6E1 = "V6E-1"


class GetAssignmentResponse(BaseModel):
    xsrf_token: str = Field(..., alias="xsrfToken")


class Outcome(str, Enum):
    SUCCESS = "SUCCESS"
    DENYLISTED = "DENYLISTED"
    QUOTA_DENIED_REQUESTED_VARIANTS = "QUOTA_DENIED_REQUESTED_VARIANTS"
    QUOTA_EXCEEDED_USAGE_TIME = "QUOTA_EXCEEDED_USAGE_TIME"
    UNDEFINED_OUTCOME = "UNDEFINED_OUTCOME"


class RuntimeProxyInfo(BaseModel):
    token: str
    token_expires_in_seconds: int = Field(..., alias="tokenExpiresInSeconds")
    url: str


class Variant(str, Enum):
    DEFAULT = "VARIANT_DEFAULT"
    GPU = "VARIANT_GPU"
    TPU = "VARIANT_TPU"


class Shape(str, Enum):
    STANDARD = "SHAPE_STANDARD"
    HIGH_RAM = "SHAPE_HIGH_RAM"


class SubscriptionTier(str, Enum):
    NONE = "SUBSCRIPTION_TIER_NONE"
    PAY_AS_YOU_GO = "SUBSCRIPTION_TIER_PAY_AS_YOU_GO"
    COLAB_PRO = "SUBSCRIPTION_TIER_PRO"
    COLAB_PRO_PLUS = "SUBSCRIPTION_TIER_PRO_PLUS"


class SubscriptionState(str, Enum):
    SUBSCRIBED = "SUBSCRIBED"
    UNSUBSCRIBED = "UNSUBSCRIBED"


class CcuInfo(BaseModel):
    current_balance: float = Field(..., alias="currentBalance")
    consumption_rate_hourly: float = Field(..., alias="consumptionRateHourly")
    assignments_count: int = Field(..., alias="assignmentsCount")


class UserInfo(BaseModel):
    subscription_tier: SubscriptionTier = Field(..., alias="subscriptionTier")


class Assignment(BaseModel):
    accelerator: Accelerator
    endpoint: str
    idle_timeout_sec: int = Field(..., alias="idleTimeoutSec")
    machine_shape: Shape = Field(..., alias="machineShape")
    runtime_proxy_info: RuntimeProxyInfo = Field(..., alias="runtimeProxyInfo")
    subscription_state: SubscriptionState = Field(..., alias="subscriptionState")
    subscription_tier: SubscriptionTier = Field(..., alias="subscriptionTier")
    variant: Variant


class PostAssignmentResponse(Assignment):
    outcome: Optional[Outcome] = None


XSSI_PREFIX = ")]}'\n"
TUN_ENDPOINT = "/tun/m"


class InvalidSchemaError(Exception):
    """Raised if the given schema for the request is invalid/missing."""


class ListedAssignment(BaseModel):
    accelerator: Accelerator
    endpoint: str
    variant: Variant
    machine_shape: Shape = Field(..., alias="machineShape")


class ListedAssignments(BaseModel):
    assignments: List[ListedAssignment]


class ColabRequestError(Exception):
    def __init__(self, message, request, response, response_body=None):
        super().__init__(message)
        self.request = request
        self.response = response
        self.response_body = response_body


class TooManyAssignmentsError(Exception):
    pass


class DenylistedError(Exception):
    pass


class InsufficientQuotaError(Exception):
    pass


class ColabClient:
    def __init__(
        self, colab_domain: str, colab_api_domain: str, get_access_token, logger=None
    ):
        self.colab_domain = colab_domain
        self.colab_api_domain = colab_api_domain
        self.get_access_token = get_access_token
        self.session = requests.Session()
        if "localhost" in self.colab_domain:
            self.session.verify = False
        self.logger = logger or logging.getLogger(__name__)

    def _strip_xssi_prefix(self, v: str) -> str:
        if not v.startswith(XSSI_PREFIX):
            self.logger.debug(f"XSSI prefix not found in response: {v}")
            return v
        stripped_v = v[len(XSSI_PREFIX) :]
        self.logger.debug(f"Stripped XSSI prefix, returning: {stripped_v}")
        return stripped_v

    def _issue_request(
        self,
        endpoint: str,
        method: str = "GET",
        headers: Dict[str, str] = None,
        params: Dict[str, str] = None,
        schema: Optional[BaseModel] = None,
        **kwargs,
    ):
        if not schema:
            raise InvalidSchemaError()

        parsed_endpoint = urlparse(endpoint)
        if parsed_endpoint.hostname in urlparse(self.colab_domain).hostname:
            if params is None:
                params = {}
            params["authuser"] = "0"

        token = self.get_access_token()
        request_headers = headers.copy() if headers else {}
        request_headers[ACCEPT_JSON_HEADER["key"]] = ACCEPT_JSON_HEADER["value"]
        request_headers[AUTHORIZATION_HEADER["key"]] = f"Bearer {token}"
        request_headers[COLAB_CLIENT_AGENT_HEADER["key"]] = COLAB_CLIENT_AGENT_HEADER[
            "value"
        ]

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Request: {method} {endpoint}")
            self.logger.debug(f"Headers: {request_headers}")
            self.logger.debug(f"Params: {params}")

        response = self.session.request(
            method, endpoint, headers=request_headers, params=params, **kwargs
        )

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(f"Response: {response.status_code} {response.reason}")
            self.logger.debug(f"Response Body: {response.text}")

        if not response.ok:
            raise ColabRequestError(
                f"Failed to issue request {method} {endpoint}: {response.reason}",
                request=response.request,
                response=response,
                response_body=response.text,
            )

        body = self._strip_xssi_prefix(response.text)
        if not body:
            return
        return TypeAdapter(schema).validate_python(json.loads(body))

    def get_subscription_tier(self) -> SubscriptionTier:
        url = urljoin(self.colab_api_domain, "v1/user-info")
        user_info = self._issue_request(url, schema=UserInfo)
        return user_info.subscription_tier

    def get_ccu_info(self) -> CcuInfo:
        url = urljoin(self.colab_domain, f"{TUN_ENDPOINT}/ccu-info")
        return self._issue_request(url, schema=CcuInfo)

    def list_assignments(self) -> List[ListedAssignment]:
        url = urljoin(self.colab_domain, f"{TUN_ENDPOINT}/assignments")
        assignments = self._issue_request(url, schema=ListedAssignments)
        return assignments.assignments

    def assign(
        self,
        notebook_hash: uuid.UUID,
        variant: Variant,
        accelerator: Optional[Accelerator] = None,
    ) -> Dict[str, Any]:
        assignment = self._get_assignment(notebook_hash, variant, accelerator)
        if isinstance(assignment, Assignment):
            return {"assignment": assignment, "is_new": False}

        try:
            res = self._post_assignment(
                notebook_hash, assignment.xsrf_token, variant, accelerator
            )
        except ColabRequestError as e:
            if e.response.status_code == 412:
                raise TooManyAssignmentsError(str(e))
            raise e

        if res.outcome in [
            Outcome.QUOTA_DENIED_REQUESTED_VARIANTS,
            Outcome.QUOTA_EXCEEDED_USAGE_TIME,
        ]:
            raise InsufficientQuotaError(
                "You have insufficient quota to assign this server."
            )
        if res.outcome == Outcome.DENYLISTED:
            raise DenylistedError(
                "This account has been blocked from accessing Colab servers."
            )

        return {"assignment": res, "is_new": True}

    def _build_assign_url(
        self,
        notebook_hash: uuid.UUID,
        variant: Variant,
        accelerator: Optional[Accelerator] = None,
    ) -> str:
        url = urljoin(self.colab_domain, f"{TUN_ENDPOINT}/assign")
        params = {"nbh": uuid_to_web_safe_base64(notebook_hash)}
        if variant != Variant.DEFAULT:
            params["variant"] = variant.value
        if accelerator:
            params["accelerator"] = accelerator.value

        req = requests.Request("GET", url, params=params)
        prep = self.session.prepare_request(req)
        return prep.url

    def _get_assignment(
        self,
        notebook_hash: uuid.UUID,
        variant: Variant,
        accelerator: Optional[Accelerator] = None,
    ) -> Union[GetAssignmentResponse, Assignment]:
        url = self._build_assign_url(notebook_hash, variant, accelerator)

        # A bit of a hack to handle union types with pydantic
        try:
            return self._issue_request(url, schema=Assignment)
        except Exception:
            return self._issue_request(url, schema=GetAssignmentResponse)

    def _post_assignment(
        self,
        notebook_hash: uuid.UUID,
        xsrf_token: str,
        variant: Variant,
        accelerator: Optional[Accelerator] = None,
    ) -> PostAssignmentResponse:
        url = self._build_assign_url(notebook_hash, variant, accelerator)
        headers = {COLAB_XSRF_TOKEN_HEADER["key"]: xsrf_token}
        return self._issue_request(
            url, method="POST", headers=headers, schema=PostAssignmentResponse
        )
