"""Campaign CRUD routes."""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_session
from models import Campaign
from schemas import CampaignCreate, CampaignRead, CampaignUpdate

router = APIRouter(prefix="/campaigns", tags=["campaigns"])


@router.post("/", response_model=CampaignRead, status_code=201)
async def create_campaign(
    data: CampaignCreate,
    session: AsyncSession = Depends(get_session),
):
    campaign = Campaign(**data.model_dump())
    session.add(campaign)
    await session.commit()
    await session.refresh(campaign)
    return campaign


@router.get("/", response_model=list[CampaignRead])
async def list_campaigns(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Campaign).order_by(Campaign.name))
    return result.scalars().all()


@router.get("/{campaign_id}", response_model=CampaignRead)
async def get_campaign(
    campaign_id: int,
    session: AsyncSession = Depends(get_session),
):
    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    return campaign


@router.patch("/{campaign_id}", response_model=CampaignRead)
async def update_campaign(
    campaign_id: int,
    data: CampaignUpdate,
    session: AsyncSession = Depends(get_session),
):
    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(campaign, key, value)
    await session.commit()
    await session.refresh(campaign)
    return campaign


@router.delete("/{campaign_id}", status_code=204)
async def delete_campaign(
    campaign_id: int,
    session: AsyncSession = Depends(get_session),
):
    campaign = await session.get(Campaign, campaign_id)
    if not campaign:
        raise HTTPException(404, "Campaign not found")
    await session.delete(campaign)
    await session.commit()
