import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)


register(
    id='Delivery-v0',
    entry_point='gym_delivery.envs:DeliveryEnv',
    timestep_limit=1000,
)
