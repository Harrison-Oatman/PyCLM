# Setting up with an AI agent

PyCLM comes with three *agent skills*: step-by-step guides that a coding
agent such as [Claude Code](https://claude.com/claude-code) follows to walk
you through the slow parts. The agent runs the `pyclm` checks itself and asks
you only about your microscope and your experiment.

- [`pyclm-microscope-bringup`](https://github.com/Harrison-Oatman/PyCLM/blob/main/.claude/skills/pyclm-microscope-bringup/SKILL.md):
  installing PyCLM and getting it running on your microscope (the
  Micro-Manager device interface, presets, `pyclm_config.toml`, the first
  real run).
- [`pyclm-design-experiment`](https://github.com/Harrison-Oatman/PyCLM/blob/main/.claude/skills/pyclm-design-experiment/SKILL.md):
  turning an experiment idea into a checked experiment folder and a dry run.
- [`pyclm-write-pattern-method`](https://github.com/Harrison-Oatman/PyCLM/blob/main/.claude/skills/pyclm-write-pattern-method/SKILL.md):
  turning a light rule described in words into a tested pattern method.

## Using them with Claude Code

1. Install Claude Code ([instructions](https://code.claude.com/docs/en/quickstart)).
2. Clone PyCLM and start Claude Code in it:

   ```bash
   git clone https://github.com/Harrison-Oatman/PyCLM.git
   cd PyCLM
   claude
   ```

3. Describe what you want, for example *"help me set up PyCLM on our Nikon
   Ti2 with a Mightex Polygon"* or *"a bar of light moving at 1 µm/min over
   half my wells"*. The agent picks the matching skill.

To use the skills from another folder, such as the one holding your
experiments, copy the folders in `.claude/skills/` into that folder's
`.claude/skills/`, or into `~/.claude/skills/` to have them everywhere. The
skills are plain Markdown, so other agents, or a person, can follow them too.

The agent never changes your Micro-Manager configuration; it tells you what
to change. Its work on the virtual microscope (`--dry`) is safe to try before
going to the scope.
