#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  Trial and Error Agent for L2RPN Challenge 2022
#
#  Copyright 2022 Honda Research Institute Europe GmbH
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions are
#  met:
#
#  1. Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
#
#  2. Redistributions in binary form must reproduce the above copyright
#     notice, this list of conditions and the following disclaimer in the
#     documentation and/or other materials provided with the distribution.
#
#  3. Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
#  IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
#  THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
#  LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
#  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
#  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#


from grid2op.Agent import BaseAgent
import numpy as np


class TAEAgent(BaseAgent):
    """
    Trial and Error Agent
    Determines curtailment, dispatch and discharge of stationary batteries through random search
    and searches for lines to cut via brute-force approach.
    """

    def __init__(self, env):
        """Initialize a new agent."""
        BaseAgent.__init__(self, action_space=env.action_space)
        self.env = env
        # fixed seed for random actions in order to generate reproducible challenge results
        np.random.seed(42)

    def optimize_with_trail_and_error(self, observation, ref_act=None):
        """Trial and error optimization of actions"""

        # divide generators in two groups (redispatchable and curtailable)
        dispatchable_units = []
        curtail_units = []
        upper_bound = []
        lower_bound = []
        for i, redispatchable in enumerate(self.env.gen_redispatchable):
            if redispatchable:
                dispatchable_units.append(i)
                upper_bound.append(self.env.gen_max_ramp_up[i])
                lower_bound.append(-self.env.gen_max_ramp_down[i])
            else:
                curtail_units.append(i)

        # create empty reference action object if no reference action object is given
        if ref_act is None:
            ref_act = self.env.action_space()  # do-nothing action

        # check performance of reference action
        act = ref_act.copy()
        best_rho = 1000
        best_dispatch = None
        best_curtail = None
        best_discharge = None
        (
            state_after_simulate,
            _,
            simulated_done,
            _,
        ) = observation.simulate(act, time_step=1)
        if not simulated_done:
            max_rho = max(state_after_simulate.rho)
            if max_rho < best_rho:
                best_dispatch = []
                best_curtail = []
                best_discharge = []
                best_rho = max_rho
            print("Predicted max rho of " + str(max_rho) + " with reference action (or do-nothing)")
        else:
            print("Predicted Game Over with reference action (do-nothing)")

        # try 1000 random actions (redispatch, curtail and battery usage)
        n_iterations = 1
        while n_iterations < 1000 and best_rho > 0.8:
            # create random action (dispatch, curtail, discharge)
            dispatch = np.random.uniform(lower_bound, upper_bound, len(dispatchable_units))
            dispatch_act = [(dispatchable_units[i], dispatch[i]) for i in range(len(dispatchable_units))]
            curtail = np.random.uniform(0.0, 1.0, len(curtail_units))
            curtail_act = [(curtail_units[i], curtail[i]) for i in range(len(curtail_units))]
            p_max = self.env.storage_max_p_prod
            discharge = np.random.uniform(0.0, p_max, len(p_max))
            discharge_act = [(i, -discharge[i]) for i in range(len(p_max))]
            act.redispatch = dispatch_act
            act.curtail = curtail_act
            act.storage_p = discharge_act

            # test random action
            (
                state_after_simulate,
                _,
                simulated_done,
                _,
            ) = observation.simulate(act, time_step=1)

            # remember best random action
            if not simulated_done:
                max_rho = max(state_after_simulate.rho)
                if max_rho < best_rho:
                    best_dispatch = dispatch_act
                    best_curtail = curtail_act
                    best_discharge = discharge_act
                    best_rho = max_rho
            n_iterations += 1

        # recreate best action for return
        if best_dispatch is not None:
            print("Best Action rho: " + str(best_rho))
            act = ref_act.copy()
            # overwrite reference action for redispatch, curtail and storage usage
            act.redispatch = best_dispatch
            act.curtail = best_curtail
            act.storage_p = best_discharge
            return act

        # in case all random actions are worse than reference action, return reference action unchanged
        print("No successful action found")
        return ref_act.copy()

    def simulate_line_toggle(self, observation, line_id):
        """Simulate a single line toggle one time step ahead"""
        act = self.env.action_space()
        toggle_status = np.zeros(len(observation.line_status)) > 0
        toggle_status[line_id] = True
        act.line_change_status = toggle_status
        (
            state_after_simulate,
            _,
            simulated_done,
            _,
        ) = observation.simulate(act, time_step=1)
        max_rho = max(state_after_simulate.rho)

        return max_rho, simulated_done

    def cut_the_line(self, observation):
        """Trial and error line cutting to improve energy flow"""

        # check performance of not acting
        no_act = self.env.action_space()  # do-nothing action
        (
            state_after_simulate,
            _,
            simulated_done,
            _,
        ) = observation.simulate(no_act, time_step=1)
        max_rho_no_act = max(state_after_simulate.rho)

        best_rho = max_rho_no_act
        best_id = -1
        # simulate single cut of all available lines
        for line_id in range(0, len(observation.line_status)):
            max_rho, simulated_done = self.simulate_line_toggle(observation, line_id)
            print(f"cut {line_id} --> rho: {max_rho} {simulated_done}")
            # remember best line cut
            if max_rho < best_rho and not simulated_done:
                best_rho = max_rho
                best_id = line_id

        # return best line cut action if it is better than doing nothing
        if best_rho < max_rho_no_act:
            act = self.env.action_space()
            toggle_status = np.zeros(len(observation.line_status)) > 0
            toggle_status[best_id] = True
            act.line_change_status = toggle_status
            return act

        # line cut is worse than no acting --> return no action
        return no_act

    def act(self, observation, reward, done=False):
        """The action that your agent will choose depending on the observation, the reward, and whether the state is
        terminal"""

        # step 1: try to reconnect lines
        best_rho = 1.0
        best_id = -1
        if np.invert(observation.line_status).any():
            for line_id in np.where(observation.line_status is False)[0]:
                # do not try to reconnect lines during cool-down
                if observation.time_before_cooldown_line[line_id] > 0:
                    continue
                max_rho, simulated_done = self.simulate_line_toggle(observation, line_id)

                print(f"recon {line_id} --> rho: {max_rho} {simulated_done}")
                if max_rho < best_rho and not simulated_done:
                    best_rho = max_rho
                    best_id = line_id

        # if line reconnect does not create a critical line status elsewhere, then reconnect best line found
        if best_rho < 1.0:
            act = self.env.action_space()
            toggle_status = np.zeros(len(observation.line_status)) > 0
            toggle_status[best_id] = True
            act.line_change_status = toggle_status
            return act

        # step 2: charge batteries if no line failure is imminent
        no_act = self.env.action_space()  # do-nothing action
        (
            state_after_simulate,
            _,
            simulated_done,
            _,
        ) = observation.simulate(no_act, time_step=1)
        max_rho_no_act = max(state_after_simulate.rho)

        if not simulated_done and max_rho_no_act < 1.0:
            # try to charge batteries
            p_max = self.env.storage_max_p_absorb
            charge_act = [(i, p_max[i] / 2.0) for i in range(len(p_max))]
            act = self.env.action_space()
            act.storage_p = charge_act
            (
                state_after_simulate,
                _,
                simulated_done,
                _,
            ) = observation.simulate(act, time_step=1)
            max_rho = max(state_after_simulate.rho)
            if not simulated_done and max_rho < 1.0:
                return act

            return no_act

        # step 3: line failure imminent, try to find countermeasure by sampling and simulating random actions
        act = self.optimize_with_trail_and_error(observation)
        (
            state_after_simulate,
            _,
            simulated_done,
            _,
        ) = observation.simulate(act, time_step=1)
        max_rho = max(state_after_simulate.rho)

        # perform found action if it prevents line failure
        if max_rho < 1.0:
            return act

        # step 4: random actions did not succeed in preventing line failure --> try to cut lines to redirect energy
        print("TRY TO CUT")
        act_cut = self.cut_the_line(observation)
        act_cut = self.optimize_with_trail_and_error(observation, act_cut)
        (
            state_after_simulate,
            _,
            simulated_done,
            _,
        ) = observation.simulate(act_cut, time_step=1)
        max_rho_cut = max(state_after_simulate.rho)

        # perform line cut if it prevents line failure
        if max_rho_cut < max_rho:
            return act_cut

        # line cut did prevent line failure --> use best random action and hope to solve problem in next time step
        return act


def make_agent(env, this_directory_path):
    """Challenge defined interface"""
    my_agent = TAEAgent(env)
    return my_agent
