# EPFL-Smart-Kitchen

Understanding behavior requires datasets that capture humans while carrying out complex tasks. The kitchen is an excellent environment for assessing human motor and cognitive function, as many complex actions are naturally exhibited in kitchens, from chopping to cleaning. Here, we introduce the EPFL-Smart-Kitchen-30 dataset, collected in a non-invasive motion capture platform inside a kitchen environment. Nine static RGB-D cameras, inertial measurement units (IMUs), and one head-mounted HoloLens~2 headset were used to capture 3D hand, body, and eye movements. 

The EPFL-Smart-Kitchen-30 dataset is a multi-view action dataset Fith synchronized exocentric, egocentric, depth, IMUs, eye gaze, body and hand kinematics spanning 29.7 hours of 16 subjects cooking four different recipes. Action sequences were densely annotated with 33.78 action segments per minute. Leveraging this multi-modal dataset, we propose four benchmarks to advance behavior understanding and modeling through:
1) a vision-language benchmark,
2) a semantic text-to-motion generation benchmark,
3) a multi-modal action recognition benchmark,
4) a pose-based action segmentation benchmark.
 
Check out the data [annotation/pose](https://zenodo.org/records/15551913) and [video](https://zenodo.org/records/15535461) data on Zenodo! We also share Lemonade on Huggingface, and you can easily use it via x. 

## Reference 

Check out our preprint for more [details](https://arxiv.org/abs/2506.01608)!

```bibtex
@article{bonnetto2025epfl,
  title={EPFL-Smart-Kitchen-30: Densely annotated cooking dataset with 3D kinematics to challenge video and language models},
  author={Bonnetto, Andy and Qi, Haozhe and Leong, Franklin and Tashkovska, Matea and Rad, Mahdi and Shokur, Solaiman and Hummel, Friedhelm and Micera, Silvestro and Pollefeys, Marc and Mathis, Alexander},
  journal={arXiv preprint arXiv:2506.01608},
  year={2025}
}
```

## Acknowledgements

We thank members of the Mathis Group for Computational Neuroscience & AI (EPFL) for their feedback throughout the project. This work was funded by EPFL, Swiss SNF grant (320030-227871), Microsoft Swiss Joint Research Center, and a Boehringer Ingelheim Fonds PhD stipend (H.Q.). We are grateful to the Brain Mind Institute for providing funds for hardware and to the Neuro-X Institute for providing funds for services.
