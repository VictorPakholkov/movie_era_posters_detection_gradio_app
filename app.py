from fastai import *
from fastai.vision.all import *
learn = load_learner('export.pkl')
labels = learn.dls.vocab
def predict(img):
    img = PILImage.create(img)
    pred,pred_idx,probs = learn.predict(img)
    return {labels[i]: float(probs[i]) for i in range(len(labels))}
import skimage
title = "Movie Poster Era Classifier"
description = "This is a study project for models deployment via Gradio and HuggingFace Spaces. The app takes a movie poster as an input, returning probabilities of the poster being drawn either in 1900s-1960s, 1970s-1990s or later on. Pretrained resnet 101 was taken as a learner with being fine-tuned on 1500 posters of each period, retreived from DuckDuckGo search queries. Quality of the resulting model is quite low with 0.35 error rate after 11 epochs, but the aim was more to play around with Gradio, rather than crating working movie poster classifier. Full code for the model and deployment is featured on my GitHub page, linked below"
examples = ['example_1.jpeg', 'example_2.jpeg', 'example_3.jpeg']
article="<p style='text-align: center'><a href='https://github.com/VictorPakholkov/movie_era_posters_detection_gradio_app' target='_blank'>Github repo</a></p>"
interpretation='default'
enable_queue=True
import gradio as gr
gr.Interface(fn=predict,
              inputs=gr.inputs.Image(shape=(512, 512)),
              outputs=gr.outputs.Label(num_top_classes=3),
              title=title,
              description=description,
              article=article,
              examples=examples,
              interpretation=interpretation,
              enable_queue=enable_queue).launch(share=True)

