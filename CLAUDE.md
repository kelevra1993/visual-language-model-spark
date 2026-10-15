<!-- Global rules go here -->
# General Coding Rules For All Projects : GUIDELINES

## 1. Critical Rules

> [!CRITICAL]
> **STRICT COMMIT POLICY**
> The AI agent is STRICTLY FORBIDDEN from executing `git commit` or `git push` unless the user explicitly requests it in their current prompt (e.g., "commit and push this"). Do not assume a finished task implies consent to commit. Violation of this is a critical failure.
> Furthermore, when the user requests to push a specific file, the AI agent MUST ONLY add, commit, and push that explicit file, and MUST NOT bundle it with other modified files, regardless of dependencies.

> [!CRITICAL]
> **TESTING REFERENCE FILES**
> The AI agent is STRICTLY FORBIDDEN from generating, saving, or committing `.pt` reference files used for testing without explicit confirmation from the user. These files act as the ground truth and modifying them automatically could hide regressions.



## 2. Development & Workflow Standards
- **Git Commit Messages**: You MUST use the standard project format: `[TAG] Detailed description ending with a period.`
  - Examples: `[Kubernetes] Complete Triton deployment configuration.`, `[INFRA] Move Terraform cluster to europe-west1.`, `[DOCS] Add terraform auto-approve flag.`
- **YAML Commenting**: Maintain the highly descriptive, instructional commenting style.

## 3. General Structure
- Keep code separated into appropriate directories (`app/loss`, `app/utilities`, `app/trainer`, etc.) for clear separation of concerns.
- When writing debugging scripts that will be kept in the repository, they should go in the scripts folder, with a structure mimicking the app folder, for easier retrieval.

## 4. Python Type Hinting
- **Type Hints**: Functions should have their arguments with the type for the arguments.
- **Return Types**: Functions must explicitly declare their return types in the function definition signature (e.g., `def my_function() -> str:`).

## 5. Naming Conventions
- **Variable Names**: Variable names have to be spelled out properly. No cryptic name variables. Everything should be easily understandable and readable.
  - *Examples (this list is not exhaustive, apply this logic universally):*
    - Avoid `_num` ➔ Use `_number`
    - Avoid `_gt` ➔ Use `_ground_truth`
    - Avoid `_val` ➔ Use `_validation` (or `_value`)
    - Avoid `_idx` or `i` ➔ Use `_index`
    - Avoid `_tmp` ➔ Use `_temporary`
    - Avoid `_curr` ➔ Use `_current`
    - Avoid `_cfg` ➔ Use `_configuration`
    - Avoid `_dict` ➔ Use `_dictionary`

## 6. Python Docstrings
- **Docstring Style**: Functions should have the "Args" docstring type and not ":param" types.
- **Function Context**: Functions must always start with a brief explanation of their role and context within the whole project pipeline. Avoid bland, generic descriptions (e.g., instead of "This function processes the image", explain *why* it processes the image and for *what* downstream service in the project pipeline).
- **Module Docstring**: Every Python script must start, at the very top of the file and before the imports, with a short module docstring written with `""" """` quotes. It describes what the file does or contains and gives the context of how it fits into the bigger picture of the project (which pipeline step it belongs to, what feeds it and what consumes its output). Avoid bland descriptions such as "Utility functions".

## 7. Functions
- **Function Definitions**: Functions should not be defined inside other functions
- **Function Calling**: During function calls to avoid ambiguity call arguments explicitly like this """visualize_focal_loss(parameter_dictionary=xxx, window_size=xxx, spacing_x=xxx, spacing_y=xxx ....)""" this is done in order to quickly make sure that functions are always properly called. This is always especially important for function that were defined in the repository.

## 8. README
- **Simple Descriptions**: While writing the README.md, do not use any type of praise of the repository, keep it simple and instructional.
- **Table Of Content**: Always start with the table of contents, followed by the description of the project, and the pre-requisite. The user will then define the following sections.
- **Image Commiting**: During README.md writing, there might be images that are associated that are shown in the README.md, these images will always be in a folder called README and should be pushed once used.


## 9. Other
- **Formatting**: Try to put code on one line or just a couple of lines when possible. Group arguments together rather than putting each argument on its own line (e.g., `layers.append(nn.Conv2d(in_channels=current_input_channels, out_channels=output_channels, kernel_size=kernel_size...))`), while still explicitly naming the arguments.
  *Example of what to avoid:*
  ```python
  layers.append(
      nn.Conv2d(
          in_channels=current_input_channels,
          out_channels=output_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          bias=bias
      )
  )
  ```
- **Inline Comments**: Use highly descriptive, instructional inline comments to explain blocks of implementation logic. Do not use numbers for these inline comments.
  *Example of expected commenting style:*
  ```python
  # Iterate through each downsampling block to simulate the feature extraction path
  for index, block in enumerate(encoder_blocks):
      # ... [processing logic] ...

      # Apply max pooling to halve spatial dimensions, except for the final bottleneck layer
      if index < len(encoder_blocks) - 1:
          # ... [pooling logic] ...

  # Iterate through the decoder blocks to simulate the upsampling and reconstruction path
  for index in range(len(decoder_blocks)):
      # Upsample the current feature map and retrieve the corresponding skip connection
      # ... [upsampling logic] ...

      # Concatenate the upsampled features with the skip connection to provide high-resolution details
      # ... [concatenation logic] ...
  ```
- **Inline Comments**: As a rule of thumb avoid large streches of code where there are not comments, so for instance that might be a guiding principle. Always keep them concise, no need for sycophancy, just simple and bland.
- **Imports**: Import should be defined at the very top.

## 10. Verification & Review
- **Self-Review**: After every code change, this file must be re-read to verify that all guidelines were properly implemented before concluding the task.
- **Targeted Replacement**: Instead of globally replacing substrings (which recklessly alters indentation for identical lines in other classes or contexts), use targeted block replacements isolating the exact methods being modified.
- **Clean-Up**: Use sed whenever possible, but when using python files for modifications, do not forget to clean them up regularily.

## 11. Pre-Implementation Workflow
- **Research First**: Before modifying any code, the agent MUST first explore and read all relevant files to build a complete understanding of the affected area. Never start editing based on assumptions alone.
- **Summarize Current Behavior**: After reading, explicitly summarize how the current code works in the area being changed. This proves understanding and prevents incorrect modifications.
- **List All Changes**: Before writing any code, output a numbered list of every file and method that needs to change, and what each change will do. This acts as a plan and prevents missed updates or orphaned references.
  *Example format:*
  ```
  Current behavior:
  - [Brief description of how the code currently works]

  Needed changes:
  1. [File/location]: [What to change and why]
  2. [File/location]: [What to change and why]
  3. ...
  ```
- **Verify Completeness**: After implementing, search the codebase (e.g., `grep`) for any remaining references to old code (renamed variables, removed config keys, etc.) to ensure nothing was missed.

## 12. Planning & Artifacts
- **Implementation Plans**: Do not use separate UI artifact files for planning. Instead, write any Implementation Plan directly into the `README.md` file in a dedicated section titled `## Implementation Plan` placed at the very bottom of the file.
  *Rationale: This ensures the plan is fully visible within the repository structure, allowing the user to easily track, review, and collaborate on the plan alongside the code, rather than it being hidden in internal, agent-specific artifacts.*
- **Plan Cleanup**: Once the implementation task is fully completed and verified, the `## Implementation Plan` section must be completely deleted from the `README.md` file.
  *Rationale: This keeps the repository documentation clean and focused only on the final state of the project, removing temporary planning clutter once it has served its purpose.*

## 13. Mandatory Compliance Checklist
- **Post-Change Checklist**: After every code change, the agent MUST output a compliance checklist covering every rule in this file. Use ✅ for rules that were followed, and ❌ for rules that were violated or not applicable (with a brief justification). This checklist acts as a forcing function to ensure no rule is silently ignored.
- **Format**: The checklist must follow this exact format:
  ```
  ### Compliance Checklist
  1. ✅ **Commit Policy** — No git commit/push executed (not requested).
  2. ✅ **Testing Reference Files** — No .pt reference files generated.
  3. ✅ **YAML Commenting** — Highly descriptive, instructional commenting maintained.
  4. ✅ **Type Hints** — All function arguments and return types are annotated.
  5. ✅ **Naming Conventions** — All variable names are fully spelled out (e.g., `validation_interval_count`, not `val_count`).
  6. ✅ **Docstrings** — "Args" style used, context-aware descriptions provided.
  7. ✅ **Functions** — No nested definitions; all calls use explicit keyword arguments.
  8. ✅ **Formatting** — Arguments grouped on reasonable lines, not one-per-line.
  9. ✅ **Inline Comments** — Descriptive comments on all code blocks, no large uncommented stretches.
  10. ✅ **Imports** — All imports defined at the top of the file.
  11. ✅ **Targeted Replacement** — Used targeted block replacements, no global substring replacements.
  12. ✅ **Clean-Up** — Temporary scripts/files cleaned up.
  13. ✅ **Module Docstring** — Every Python file starts with a `""" """` docstring explaining what it does and its place in the project.
  ```
- **Non-Compliance**: If any rule was violated, the agent must use ❌, explain what went wrong, and immediately fix the violation before concluding the task
