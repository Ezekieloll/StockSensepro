# test_adversarial.py
import torch
from adversarial.scenario_simulator import ScenarioSimulator
from forecasting.lstm_model import LSTMBaseline

model = LSTMBaseline(input_size=9)
x = torch.rand(8, 17, 9)  # batch, window, features

sim = ScenarioSimulator(model)

print("Baseline:", sim.baseline(x).shape)
print("Spike:", sim.demand_spike(x).shape)
print("Drop:", sim.demand_drop(x).shape)
print("Holiday:", sim.holiday_shock(x).shape)
