#!/usr/bin/env python3

from omegaconf import DictConfig
from huggingface_hub import snapshot_download
from lerobot.common.policies.factory import make_policy, Policy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.envs.utils import preprocess_observation
from lerobot.common.envs.factory import make_env
from lerobot.common.utils.utils import init_hydra_config
import rospy
from fbot_robot_learning.srv import GetAvailablePolicies, GetAvailablePoliciesRequest, GetAvailablePoliciesResponse
from fbot_robot_learning.srv import ExecutePolicy, ExecutePolicyRequest, ExecutePolicyResponse
from fbot_robot_learning.msg import PolicyInfo
from pathlib import Path
import torch

class SkillInferenceServer:
    def __init__(self):
        self.load_parameters()
        self.policy_map = {}
        self.env_map = {}
        for policy_info in self.policy_list:
            snapshot_path = Path(snapshot_download(repo_id=policy_info['repo_id']))
            config_path = snapshot_path / "config.yaml"
            config = init_hydra_config(config_path=config_path)
            self.policy_map[policy_info['name']] = make_policy(hydra_cfg=config, pretrained_policy_name_or_path=policy_info['repo_id'])
            self.env_map[policy_info['name']] = make_env(cfg=config, n_envs=1)
        self.policy_info_service = rospy.Service('/fbot_robot_learning/get_available_policies', GetAvailablePolicies, handler=self._handle_available_policies)
        self.execute_policy_service = rospy.Service('/fbot_robot_learning/execute_policy', ExecutePolicy, handler=self._handle_execute_policy)

    def load_parameters(self):
        self.policy_list = rospy.get_param('~policies', default=[])
    
    def _handle_available_policies(self, req: GetAvailablePoliciesRequest):
        res = GetAvailablePoliciesResponse()
        for info in self.policy_list:
            policy_info = PolicyInfo()
            policy_info.name = info['name']
            policy_info.description = info['description']
            policy_info.repo_id = info['repo_id']
            policy_info.max_steps = info['max_steps']
            res.available_policies_info.append(policy_info)
        return res

    def _handle_execute_policy(self, req: ExecutePolicyRequest):
        policy: Policy = self.policy_map[req.policy_name]
        max_steps = 0
        for i in range(len(self.policy_list)):
            if self.policy_list[i]['name'] == req.policy_name:
                max_steps = self.policy_list[i]['max_steps']
                break
        device = get_device_from_parameters(policy)
        policy.reset()
        env = self.env_map[req.policy_name]
        obs, _ = env.reset()
        for i in range(max_steps):
            obs = preprocess_observation(observations=obs)
            obs = {key: obs[key].to(device, non_blocking=True) for key in obs}
            with torch.inference_mode():
                action = policy.select_action(obs)
            action = action.to('cpu').numpy()
            obs, _, _, _, _ = env.step(action)
        res = ExecutePolicyResponse()
        return res

if __name__ == '__main__':
    rospy.init_node('skill_inference_server', anonymous=True)
    server = SkillInferenceServer()
    rospy.spin()