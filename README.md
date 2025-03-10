# 🚀Audio-FLAN
**Audio-FLAN** aims to **unify audio-language models** that can seamlessly handle both **understanding** and **generation** tasks across **speech, music, sound**.

## 🔥 Audio-FLAN: A Preliminary Release
<div>
    <a href="https://huggingface.co/datasets/HKUSTAudio/Audio-FLAN-Dataset"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Huaging_Face-HKUST_Audio-yellow"></a>
    <a href="https://arxiv.org/pdf/2502.16584"><img src="https://img.shields.io/badge/arxiv-Audio_FLAN-green">
</a>
</div>
<br>

 **An Instruction-Tuning Dataset for Unified Audio Understanding and Generation Across Speech, Music, and Sound**


Recent advancements in audio tokenization have significantly enhanced the integration of audio capabilities into large language models (LLMs). However, audio understanding and generation are often treated as distinct tasks, hindering the development of truly **unified** audio-language models. While instruction tuning has demonstrated remarkable success in improving generalization and zero-shot learning across text and vision, its application to audio remains largely unexplored. A major obstacle is the lack of comprehensive datasets that unify audio understanding and generation. To address this, we introduce Audio-FLAN, a large-scale instruction-tuning dataset covering **80 diverse tasks** across **speech, music, and sound** domains, with **over 100 million instances**. **Audio-FLAN** lays the foundation for unified audio-language models that can seamlessly handle both **understanding** (e.g., transcription, comprehension) and **generation** (e.g., speech, music, sound) tasks across a wide range of audio domains in a zero-shot manner. The **Audio-FLAN** dataset is available on [HuggingFace](https://huggingface.co/datasets/HKUSTAudio/Audio-FLAN-Dataset) and [GitHub](https://github.com/lmxue/Audio-FLAN) and will be continuously updated.

![image](https://github.com/user-attachments/assets/39cb6891-1334-42d3-8d4e-8671a478d37d)

# 📌 News  
* A subset of the dataset is now available on 🤗 [HuggingFace](https://huggingface.co/datasets/HKUSTAudio/Audio-FLAN-Dataset). The full dataset will be released progressively.

# ⚠️ Disclaimer 
The Audio-FLAN dataset is released under the same license as the original datasets used in its creation. Please refer to the respective licenses of the original datasets for usage and redistribution terms. We do not claim ownership of the original data and encourage users to comply with the licensing terms of the source materials.

<!-- ## Updates -->
# 📚 Citation 
```
@article{xue2025audio,
  title={Audio-FLAN: A Preliminary Release},
  author={Liumeng Xue, Ziya Zhou, Jiahao Pan, Zixuan Li, Shuai Fan, Yinghao Ma, Sitong Cheng, Dongchao Yang, Haohan Guo, Yujia Xiao, Xinsheng Wang, Zixuan Shen, Chuanbo Zhu, Xinshen Zhang, Tianchi Liu, Ruibin Yuan, Zeyue Tian, Haohe Liu, Emmanouil Benetos, Ge Zhang, Yike Guo, Wei Xue},
  journal={arXiv preprint arXiv:2502.16584},
  year={2025}
}
```
