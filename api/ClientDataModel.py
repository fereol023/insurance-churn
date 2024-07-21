from pydantic import BaseModel


class ClientDataM(BaseModel):
    policy_duration_v1 : int
    inception_date_year_v1: int
    inception_date_month: int
    package: str # Premium etc..
    has_additionnal_policies: bool
    discount_v1 : int 
    premium : float
    age_v1 : float
    gender : str # (M/F)
    total_claims_value_v1 : float
    total_claims_number : int
    number_of_complaints : int
