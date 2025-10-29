# Sample Embodied AI Platform

A reference platform with components for collecting data, training, evaluating, and deploying embodied AI systems on AWS. 

## What's New

* [October 2025] We added the first component that demonstrates the fine-tuning pipeline for NVIDIA Isaac GR00T vision-language-action (VLA) model via teleoperation and imitation learning, then deploying for inference on cost-effective SO-ARM100/101.

## Project goals

- **Accelerate adoption**: End-to-end reference architecture combining AWS managed services with open source, purpose-built for physical/embodied AI.
- **Lower the barrier**: Train and test in the cloud, then deploy to real robots, cost-effectively and reproducibly.
- **Move fast**: Re-train overnight in AWS as tasks and environments change.
- **Ecosystem enablement**: A practical baseline for startups and enterprises to build scalable physical AI pipelines on AWS.
- **Cloud-to-robot path**: Demonstrates integration from simulation and training to on-device inference.

## Component overview

This repository is organized into modular components. Each component has its own documentation with setup, deployment, and usage instructions.

### Available components

| Component | Path | Purpose | Docs |
| --- | --- | --- | --- |
| NVIDIA Isaac GR00T Training | `training/gr00t/` | Fine-tune NVIDIA Isaac GR00T with teleop/sim data; reproducible workflow on AWS Batch; DCV workstation for monitoring/eval | [training/gr00t/README.md](training/gr00t/README.md) |

## Roadmap

- Additional VLA backbones and training recipes
- Alternative data generation: teleop, scripted, sim-to-real augmentation, synthetic video
- More embodiments (humanoids, robotic arms, etc.)
- Serving patterns (SageMaker, EKS) and agents (Bedrock, OSS)
- Robust IoT/edge deployment (AWS IoT/Greengrass), safety/telemetry best practices

## Security

Review and run security scans before production use. See:
- Each component and its own security considerations and best practices.
- [CONTRIBUTING](CONTRIBUTING.md)

## Reporting Issues

If you notice a defect, feel free to create an [Issue](https://github.com/aws-samples/sample-embodied-ai-platform/issues).

## Contributing

Contributions are welcome. Please see [CONTRIBUTING](CONTRIBUTING.md) and [CODE_OF_CONDUCT](CODE_OF_CONDUCT.md).

## License

This project is licensed under the MIT-0 License. See [LICENSE](LICENSE).

## Acknowledgments

- AWS teams and community projects
- NVIDIA Isaac team and open-source contributors

