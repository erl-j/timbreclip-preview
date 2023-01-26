# text-driven-afx-control

## Setup

- install requirements with `pip install -r requirements.txt`
- clone DeepAFX-st with `git clone https://github.com/adobe-research/DeepAFx-ST.git`
- install dependencies for DeepAFX-st by navigating to the cloned directory and running `pip install --pre -e .`
- remove @torch.jit.script decorator in the audio processors you want to use
