# MuseQuest

General-purpose multi-modal large language models (LLMs) are widely adopted today for their ability to process and generate text, images, and other media effectively. However, these models often fall short in user-specific scenarios because they are not trained on proprietary data or tailored to domain-specific knowledge, leading to less accurate or helpful responses. To address this limitation, two widely used techniques are `Retrieval-Augmented Generation (RAG)` and `model fine-tuning`.

`RAG` dynamically retrieves relevant information from external knowledge bases, providing accurate answers without requiring the model itself to be retrained as a domain expert. This approach is cost-effective and well-suited for scenarios with frequently changing knowledge. However, for certain vision-intensive applications, such as medical imaging or nuanced artifact classification, RAG is less effective since the model lacks the capability to perform specialized visual pattern recognition, a critical requirement for these applications.

`Model fine-tuning` overcomes this limitation by further training the original base model on proprietary or domain-specific datasets. This process enables the fine-tuned model to recognize complex patterns and deliver specialized visual understanding tailored to specific user scenarios. Additionally, fine-tuned models offer faster inference speeds since all necessary knowledge is embedded directly within the model.

`MuseQuest` is a multi-modal model fine-tuning project designed primarily for educational purposes. As the name implies, `MuseQuest` emulates an intelligent museum docent with specialized knowledge about exhibits. It interacts with visitors and delivers educational content about the exhibitions. When provided with an exhibit image and a visitor's question, our "museum docent" (a fine-tuned vision LLM) generates a detailed response, including a comprehensive description of the exhibit, its historical significance, or its social impact.

This project demonstrates the end-to-end process of fine-tuning a multi-modal model tailored specifically for museum touring scenarios. It showcases the creation of a synthetic dataset from raw materials using advanced LLM APIs available on the market. Additionally, the project explores popular fine-tuning approaches, ranging from straightforward single-GPU training to more complex parallel training across multiple GPUs. Finally, the `similarity score`, a metric designed to evaluate the quality of the model's responses, has been used to assess the performance improvements of the fine-tuned model compared to the original base model.

Furthermore, the scope of the project can be extended beyond museum touring scenarios. By fine-tuning the model on different datasets, this application can be adapted to a variety of use cases, such as medical diagnostics, cultural heritage analysis, industrial quality control, and more.

## Base Model

In our project, we selected Hugging Face's [Idefics2](https://huggingface.co/blog/idefics2) as the base model. Idefics2 is an 8-billion-parameter multimodal model that integrates Google's SigLIP (Significant Image Language Image Processing) vision model with Mistral AI's Mistral 7B language model, offering robust capabilities for both text and image understanding.

At the time of writing, an upgraded version, [Idefics3](https://huggingface.co/HuggingFaceM4/Idefics3-8B-Llama3), has also been released. Idefics3 is another 8-billion-parameter multimodal model from Hugging Face, combining Google's SigLIP vision model with Meta's Llama3 8B instruct model. All the fine-tuning techniques discussed in this project are fully applicable to Idefics3 as well.

## Dataset

We focused on 65 exhibitions from the Smithsonian Institution ([si.edu](https://www.si.edu)). The raw exhibition information, including official descriptions and exhibition images, was collected from [si.edu](https://www.si.edu). Using this data, we generated a synthetic dataset with the help of the OpenAI API ([openai.com](https://www.openai.com)). Specifically, `GPT-4o mini` was utilized with three different temperature settings to generate a total of 15 question-answer pairs for each exhibition. Each question-answer pair was then combined with a corresponding exhibition image to create a complete data sample.

For exhibitions that lacked sufficient images, we leveraged the [Albumentations](https://albumentations.ai/) tool to augment new images from the original exhibition images sourced from si.edu. This ensured that every exhibition was abundantly represented in the dataset, providing a robust foundation for fine-tuning purposes.

## Fine-Tuning Approaches

In this project, we explored several popular fine-tuning approaches:

- `LoRA Fine-Tuning`: A widely used technique that delivers good fine-tuned model quality with relatively low hardware requirements. This serves as our baseline fine-tuning approach.
- `QLoRA Fine-Tuning`: Builds on LoRA by quantizing the base model, significantly reducing total GPU VRAM requirements while achieving model quality close to that of LoRA fine-tuned models.
- `LoRA Fine-Tuning with Distributed Data Parallel (DDP)`: Extends LoRA by enabling parallel training across multiple GPUs, maximizing training speed while retaining the quality of LoRA fine-tuned models.
- `Full Fine-Tuning`: Offers higher training granularity, resulting in slightly better fine-tuned model quality compared to LoRA in this project's settings.
- `Full Fine-Tuning with Model Parallel (MP)`: Reduces GPU VRAM requirements per GPU by vertically partitioning the model (splitting by layers) for parallel training across multiple GPUs, achieving model quality comparable to Full Fine-Tuning.
- `Full Fine-Tuning with Fully Shared Data Parallel (FSDP)`: Enhances training speed by horizontally sharding the model (splitting within layers) for parallel training across multiple GPUs, delivering model quality comparable to Full Fine-Tuning.

Readers can select the most suitable approach based on their specific needs. It's also important to note that these approaches are not mutually exclusive and can often be combined to achieve optimal results.

## Model Quality Comparison

#### Model Quality Comparison by Similarity Score

As shown in the figure below, the best possible [Similarity Score](./2_model_finetuning/model_finetuning_readme.md#similarity-score) is `0`, indicating that the model's generated answers are identical to the [Ground Truth](./2_model_finetuning/model_finetuning_readme.md#ground-truth).  

The `base model` (No Fine-Tuning) achieves a `Similarity Score` close to `0.6`, while `OpenAI GPT-4o` provides better answer quality, achieving a `Similarity Score` around `0.3`. `LoRA Fine-Tuning` significantly enhances the model's quality, reducing the `Similarity Score` to approximately `0.04`, while `Full Fine-Tuning` further improves it to around `0.03`.  

<img src="./assets/similarity_score_comparison.jpg" height="1500" alt="Similarity Score Comparison" style="border: 1px solid #ccc; padding: 5px; display: block;">

#### Model Quality Comparison by Individual Test Results

The table below demonstrates the quality of responses generated by different models. The fine-tuned models produce nuanced and informative answers, capturing intricate details about craftsmanship, history, and cultural significance. These responses offer greater depth and contextual richness compared to those from the `base model` and `OpenAI GPT-4o`.  

| **User Query** <br>(with an exhibition image and a question) | **Generated Answer from Base Model** | **Generated Answer from OpenAI GPT-4o** | **Generated Answer from LoRA Fine-Tuned Model** |
|--------------------------------------------------------------|--------------------------------------|-----------------------------------------|-------------------------------------------------|
|<img src="./assets/weathervane-teapot-1.jpg" width="1000" alt="Weathervane Teapot" style="border: 1px solid #ccc; padding: 5px; display: block;"> Can you describe this exhibit?<sup>[1]</sup>|This is a sculpture.|This exhibit is a sculpture composed of various geometric and abstract shapes. The structure includes a primary base with two triangular spikes. Several colorful elements are mounted on a horizontal rod, including spherical and rectangular forms. The right side features a white, funnel-like object with a curved yellow element attached. Above, there is a horizontal rod with blue triangular and rectangular shapes, giving it a kinetic feel similar to a mobile. The composition is modern and abstract, featuring a mix of materials and colors that create a dynamic and playful aesthetic.|This exhibit features the innovative work of Peter Shire, an artist born in Los Angeles in 1947. One of the standout pieces is the Weathervane Teapot, created in 1981, which exemplifies Shire's unique approach to ceramic art. The teapot is characterized by its playful design, incorporating colorful shapes such as squares and spheres, along with whimsical rods and hollow tubes. Shire's intention was to challenge traditional, class-based perceptions of "tea-time," making it a practice that is accessible to everyone. The teapot's unconventional construction invites viewers to engage with it, raising questions about its functionality and how to handle it. After 1985, Shire shifted his focus to furniture design, marking a significant transition in his artistic journey. This piece is part of the Smithsonian American Art Museum's collection, generously donated by Donna and John Donaldson in honor of the anniversaries of the James Renwick Alliance and the Renwick Gallery.|
|<img src="./assets/nakhla-meteorite-fragment-1.jpg" width="1000" alt="Nakhla Meteorite Fragment" style="border: 1px solid #ccc; padding: 5px; display: block;"> Can you describe this exhibit?<sup>[2]</sup>|This is a large rock that is textured and colored unevenly.|This image shows a piece of the Nakhla meteorite. The exhibit likely highlights its significance as a Martian meteorite, which fell in Egypt in 1911. The meteorite is notable for its smooth, fusion-crusted surface and the rough, lighter-colored interior, which is visible due to the cut or break. The Nakhla meteorite is famous for the story of hitting a dog and has been studied for clues about Mars's geology and potential past life.|This exhibit features the Nakhla Meteorite Fragment at the Smithsonian's National Museum of Natural History, a significant specimen among the thirty-four known meteorites that have originated from Mars. The Nakhla meteorite fell on June 28, 1911, in the Nakhla region of Alexandria, Egypt, where it was observed exploding in the atmosphere before breaking into approximately forty pieces, with a total weight of 22 pounds. Many of these fragments were found buried up to a meter deep. The Smithsonian acquired two samples shortly after the fall, and in 1962, it added a 480-gram fragment, bringing its total collection to 650 grams by the 1970s. The Nakhlites, a term referring to Martian meteorites named after Nakhla, are igneous rocks rich in augite formed from basaltic magma around 1.3 billion years ago. Research indicates that these meteorites were once hydrated and were altered by liquid water around 620 million years ago. They were ejected from Mars due to an asteroid impact about 10.75 million years ago and landed on Earth within the last 10,000 years.|
|<img src="./assets/plymouth-rock-piece-1.jpg" width="1000" alt="Plymouth Rock Piece" style="border: 1px solid #ccc; padding: 5px; display: block;"> Can you describe this exhibit?<sup>[3]</sup>|This is a rock.|This exhibit is a large, roughly hewn stone. It appears to have an uneven, textured surface with a grayish color, hinting at its possible geological origin. The shape is somewhat rectangular, with flat and angular sides, suggesting it may have been shaped or altered by human activity. This exhibit could be part of a geological, historical, or archaeological display, but specific details about its origin or significance would be needed for further context.|This exhibit features the Plymouth Rock Piece from 1620, an iconic artifact at the National Museum of American History. Traditionally regarded as the landing site of the Mayflower Pilgrims, Plymouth Rock is steeped in history. The rock's origins are somewhat mysterious, but it is believed to have been moved from its original location in the 1700s to its current spot on the shore of Plymouth Harbor. In 1880, a significant piece was discovered being used as a doorstep by the Plymouth Antiquarian Society, which was later broken into three parts. One of these pieces was donated to the Smithsonian in 1984, and museum officials accepted it in 1985. This rock fragment, measuring 22 inches by 11.5 inches, is composed of granite and has a fascinating history tied to the Pilgrims' arrival in Plymouth, Massachusetts.|
|<img src="./assets/1903-wright-flyer-8.jpg" width="1000" alt="Wright Flyer" style="border: 1px solid #ccc; padding: 5px; display: block;"> What can you tell me about this exhibit?<sup>[4]</sup>|A collection of old fashioned planes are on display in a museum.|This exhibit features the Wright Flyer, the airplane used by the Wright brothers for the first successful powered flights in 1903 at Kitty Hawk, North Carolina. It is displayed at the National Air and Space Museum in Washington, D.C. <br>The Wright Flyer was groundbreaking because it was the first powered, heavier-than-air machine to achieve controlled, sustained flight with a pilot aboard. The aircraft utilizes a biplane design with a wingspan of approximately 40 feet and was powered by a custom-built engine. The structure of the plane is made of wood and fabric, showcasing early aviation engineering.<br>This exhibit is significant as it marks the beginning of the modern aviation era and highlights the ingenuity and pioneering spirit of the Wright brothers.|The exhibit features the iconic 1903 Wright Flyer, a pioneering aircraft that marked the dawn of powered flight. Built by the Wright brothers, Wilbur and Orville, this biplane is powered by a 12-horsepower engine that operates two pusher propellers via a sprocket-and-chain system. Its unique landing gear consists of linear skids, and it is covered in natural fabric, without any paint or sealant. The Flyer made its historic first flight on December 17, 1903, in Kitty Hawk, North Carolina, with Orville at the controls for a brief 12-second journey. The Wrights' innovative methods, including wind tunnel testing, were instrumental in achieving this feat, establishing principles that would shape modern aeronautics. This exhibit not only highlights the Flyer as a mechanical marvel but also celebrates the Wright brothers' legacy in transforming the concept of human flight.|

[1]: Image sourced from [Smithsonian American Art Museum](http://n2t.net/ark:/65665/vk7e065d737-b5a1-4e2a-8d5a-45809fa25433).
[2]: Image sourced from [Smithsonian National Museum of Natural History](http://n2t.net/ark:/65665/376132180-df00-4bb7-82ab-e0ad54e85b41).
[3]: Image sourced from [Smithsonian National Museum of American History](https://n2t.net/ark:/65665/ng49ca746b5-2c91-704b-e053-15f76fa0b4fa).
[4]: Image sourced from [Smithsonian National Air and Space Museum](http://n2t.net/ark:/65665/nv9aa91e7c2-85b2-4904-aff0-d990694b6f1d).

For more details on model quality comparison, refer to [Model Evaluation Readme](3_model_evaluation/model_evaluation_readme.md).

## Project Architecture

### Part 1: Synthetic Data Creation

In this part of the project, as illustrated below, we created a synthetic dataset containing training, validation, and test splits from raw exhibition information (text + images) pulled from the Smithsonian Institution ([si.edu](https://www.si.edu)). For each exhibition, we created 15 data samples, and each data sample comprises a user query image, a user question, and a museum docent's response.

The user question and docent's response pairs are derived from raw exhibition text using the OpenAI API with three different temperature settings. Low-temperature-generated question-response pairs are stable, deterministic, and accurately reflect the exhibition details; therefore, low-temp pairs are treated as Ground Truth and used for testing, while high-temp pairs are used for training.

If an exhibition has few real exhibition images and cannot compose 15 data samples, augmented images are generated using the Image Augmentation Tool ([Albumentations](https://albumentations.ai/)). Since real exhibition images are more closely aligned with application scenarios, real images are prioritized for testing, while augmented versions are chosen for training.

For more detailed steps and explanations, please refer to [Synthetic Data Creation Readme](1_synthetic_data_creation/synthetic_data_creation_readme.md).

<img src="./assets/create_synthetic_dataset.png" height="400" alt="Create Synthetic Dataset" style="border: 1px solid #ccc; padding: 10px; display: block;">

### Part 2: Model Fine-Tuning

In this part of the project, as illustrated below, we covered six fine-tuning approaches in detail: `LoRA Fine-Tuning`, `QLoRA Fine-Tuning`, `LoRA Fine-Tuning with Distributed Data Parallel (DDP)`, `Full Fine-Tuning`, `Full Fine-Tuning with Model Parallel (MP)`, and `Full Fine-Tuning with Fully Shared Data Parallel (FSDP)`.

For each fine-tuning approach, we provide the following resources to help readers better understand the concept, guide them through the fine-tuning process, and assist in interpreting the fine-tuning results:
- A `readme file`: provides an overview of the fine-tuning approach.
- Experimental `Jupyter notebooks`: demonstrate the end-to-end fine-tuning process for the approach.
- `Artifacts` created during the fine-tuning process: include model information, inference outputs, similarity score comparison plots, and more.

For more details on fine-tuning settings and the results of different fine-tuning approaches, please refer to [Model Fine-Tuning Readme](2_model_finetuning/model_finetuning_readme.md).

<img src="./assets/fine_tuning_approaches.png" height="1600" alt="Fine-Tuning Approaches" style="border: 1px solid #ccc; padding: 10px; display: block;">

### Part 3: Model Evaluation

In this part of the project, we further evaluate the fine-tuned models by comparing their quality with OpenAI's `GPT-4o`, a state-of-the-art multimodal model.

During evaluation, we reuse the dataset from `Part 1: Synthetic Data Creation` and the similarity scores of the `base model`, `LoRA fine-tuned model`, and `Full fine-tuned model` from `Part 2: Model Fine-Tuning`. The primary task here is to compute similarity scores for `OpenAI GPT-4o` using responses generated from its API when provided with user queries and exhibit images.  

Finally, we plot the similarity scores of all models—`base model`, `LoRA fine-tuned model`, `Full fine-tuned model`, and `OpenAI GPT-4o`—side by side for direct comparison, using the `Ground Truth` as a reference.

For more details on model quality comparison, refer to [Model Evaluation Readme](3_model_evaluation/model_evaluation_readme.md).

## Experiment Cost  

In `Part 1: Synthetic Data Creation`, the total cost comes from OpenAI API calls to convert raw exhibition descriptions into question-answer pairs, which is negligible (less than $1). For details, please refer to the [Synthetic Data Creation Readme](1_synthetic_data_creation/synthetic_data_creation_readme.md).  

In `Part 2: Model Fine-Tuning`, the experiment cost comes from GPU rental. Running experiments for a single notebook typically costs a few dollars (less than $5). For more details on GPU rental costs and the training time of each fine-tuning approach, please refer to the [Model Fine-Tuning Readme](2_model_finetuning/model_finetuning_readme.md).  

In `Part 3: Model Evaluation`, costs include GPU rental and OpenAI API calls for generating answers to user queries based on exhibition images, both of which are negligible (less than $1).  

## References

This project greatly benefited from the open-source community. Below are the useful resources that contributed to the development of this project:

`Hugging Face Fine-Tuning Library`
- [PEFT (Parameter-Efficient Fine-Tuning) Documentation](https://huggingface.co/docs/peft/main/en/index)
- [Transformers Documentation](https://huggingface.co/docs/transformers/index)

`Hugging Face IDEFICS 2 Multi-Modal Model`
- [IDEFICS2 Model Documentation](https://huggingface.co/blog/idefics2)
- [IDEFICS2 Fine-Tuning Tutorial](https://colab.research.google.com/drive/1rm3AGquGEYXfeeizE40bbDtcWh5S4Nlq)
- [IDEFICS2 Fine-Tuning Example](https://colab.research.google.com/drive/1NtcTgRbSBKN7pYD3Vdx1j9m8pt3fhFDB)

`Ronan's (Trelis Research) YouTube Tutorials`
- [Synthetic Dataset Generation](https://youtu.be/_GkHZQYFOGM?si=tihtF6N61XDpV4Zi)
- [IDEFICS 2 Fine-Tuning](https://youtu.be/RCxa-6b9xXI?si=PiLxuzDELoMo_7Gs)
- [Multi-GPU Fine-Tuning](https://youtu.be/gXDsVcY8TXQ?si=Sw8_OIutEQAlgJ6o)

`Image Augmentation Tool`
- [Albumentations Documentation](https://albumentations.ai/)

`Original Exhibition Information for Model Fine-Tuning`
- [Smithsonian Institution](https://www.si.edu)

`OpenAI API for Synthetic Data Creation`
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference/introduction)

`Proofreading Tools`
- [ChatGPT](https://chat.openai.com/) and [Claude](https://claude.ai/) were used to improve the documentation's readability in this project.

## Copyrighted Material

This project utilizes copyrighted models, tools, and data. Detailed information about these resources, including their sources and licenses, can be found in the [CREDITS](./CREDITS.md) file.

## Ethical Use Policy

This project is released into the public domain and is free from any restrictions under [CC0](./LICENSE). However, we strongly encourage responsible and ethical use by adhering to our [Ethical Use Guidelines](./ETHICS.md) and ensuring the software is used in ways that are beneficial, fair, and aligned with community standards.
