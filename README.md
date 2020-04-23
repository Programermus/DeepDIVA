# DeepDIVA
Python Framework for Reproducible Deep Learning Experiments

## Project work - AI songwriter
This is the project of the group Deepest Learning, in the course Advanced Deep Learning - D7047E - Luleå Tekniska Universitet.
Music is something that is loved by all. Whether you love clasical music, jazz, EDM or metal, studies have found that the average westerner listens to about 25 hours of music every single week. With the advancment of AI and Deep Learning we have seen AIs that are able to do many different tasks such as image classification, speech recognition or playing even play complicated computer games like DOTA 2 at levels that surpasses, or sometimes far surpases human level performance in such tasks. Many fantastic Deep learning algorithms have also been developed for generative tasks such as generating convincing human faces, speech, art and on. So we ask ourselves the following questions. Can we use AI and Deep Learning techinques in order to write music and make it sound convincing? Can it sound like it was written by a human?

Music is a complicated discipline with many things that need to be recognized and taken into account by the AI in order to make something that sound convincing. Tempo, rhyhtm, musical key, chord progressions, cadances, melodies, song structure and so on. Music is infact not much different from language. A well written melody is much like a well written piece of text. It is not just about the notes played, in the same way that a sentence is more than the just words in the sentence. In order to extract meaning from a sentence one needs to grasp its surroudning context. Such is also the case for music. A good melody is strongly tied to the chords and rhythm underneath the melody, and often leads you from one chord to the next, and also one section to the next. Moreover we also want our AI to be able to recognize not just chords rhythm and harmony needed to create snippets of music, but also to understand song structure to write actual song. This adds further complexity to the problem.

This problem will be solved by training a Deep Learning model on midi data (basically sheet music the computer can understand) of well known songs and composers. We have scouted out a few large midi-datasets containing several hundreds of hours of music in total, for several different instruments.
### The Lakh MIDI Dataset v0.1: https://colinraffel.com/projects/lmd/
### The MAESTRO Dataset: https://magenta.tensorflow.org/datasets/maestro
### LakhNES: https://github.com/chrisdonahue/LakhNES
### Bach Doodle Dataset: https://magenta.tensorflow.org/datasets/bach-doodle

### Interesting articles
Docs: https://docs.google.com/document/d/18lyrJOOS_cc5XSmEbGuDJXQBUyLfPsb3GP4-qCOXNCs/edit?fbclid=IwAR1bqIG8we6PA4opROJ7r_EYN64TfM4uJHQc-MB5jbpTdMN1bPmmPLHMK_c
