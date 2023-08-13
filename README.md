## Movie Era Posters Detection Gradio App

This is a study project for models deployment via Gradio and HuggingFace Spaces, made for course "Practical Deep Learning for Coders" by fast.ai and the University of Queensland (https://course.fast.ai/). The app takes a movie poster (it better to be tho one of American cinema) as an input, returning probabilities of the poster being drawn either in 1900s-1960s, 1970s-1990s or later on. Pretrained resnet 101 was taken as a learner with being fine-tuned on 1500 posters of each period, retreived from DuckDuckGo search queries. Quality of the resulting model is quite low with 0.35 error rate after 11 epochs, but the aim was more to play around with Gradio, rather than crating working movie poster classifier. 

The app is hosted on HuggingFace Spaces here: https://huggingface.co/spaces/victorpakholkov/movie_era_posters_detection_gradio_app

Repository includes Jupyter notebook (movies_posters_classifier), featuring the model's fine-tuning and export. If you are to recreate the app, you'd have to export the model as '.pkl' file, adding it to your repository.

'requirments.txt' features a list of required python libraries.

'app.py' features the Gradio app itself.

Example pics are provided under corresponding files.

**Feel free to use it as an example for your own projects or basis for further developments, like fine-tuning the model further or testing your own models with it!**
