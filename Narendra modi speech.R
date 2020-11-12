####################################################################################
# Text mining and sentiment analysis on narendra modi 2020 Independence day speech #
####################################################################################

speech_data <- readLines("D:\\Data Science\\Narendra modi.txt")
View(speech_data)

#Load the Required library
library(tm)
library(readr)
library(graphics)
library(wordcloud2)
library(wordcloud)
library(rvest)
library(RWeka)
library(textstem)

class(speech_data) #character type data set

speech_data2 <- iconv(speech_data,to="UTF-8")

speech_data2 <- Corpus(VectorSource(speech_data2)) #Transform the text into vector source
class(speech_data2)
inspect(speech_data2)

speech_data2 <- tm_map(speech_data2,tolower) # Transform the uppercase data into lower case
inspect(speech_data2)

speech_data2 <- tm_map(speech_data2,removeNumbers) #Remove the numbers from the speech.
inspect(speech_data2)

speech_data2 <- tm_map(speech_data2,removeWords,stopwords("english")) #Remove all the stopwords
inspect(speech_data2)

#As this speech was given on the occasion so there are common words like independence,india
#make,made which are used frequently, so we should remove these words too.
speech_data2 <- tm_map(speech_data2,removeWords,
        c("independence","will","india","th","much","make",
          "made","also","us","can","ki"))
inspect(speech_data2)

speech_data2 <- tm_map(speech_data2,stemDocument)#generate root from the inflected word
inspect(speech_data2)

speech_data2 <- tm_map(speech_data2,lemmatize_strings)
inspect(speech_data2)

speech_data2 <- tm_map(speech_data2,removePunctuation)
inspect(speech_data2)

speech_data2 <- tm_map(speech_data2,stripWhitespace)
inspect(speech_data2)

#create a term document matrix

tdm_speech <- TermDocumentMatrix(speech_data2)
findFreqTerms(tdm_speech,lowfreq = 5) # See all the terms that have frequency more than 5

speech_matrix <- as.matrix(tdm_speech) #Transform into matrix
View(speech_matrix)

word_freq <- rowSums(speech_matrix) #sum out the total frequency of a particluar words
View(word_freq)

word_freq2 <- subset(word_freq,word_freq>10) #subset the words that have frequency more than 10 
View(word_freq2)

barplot(word_freq2,las=2,col=rainbow(7))

speech_data3 <- data.frame(speech_matrix) #Transform the matrix into data frame.
View(speech_data3)

word_freq3 <- sort(rowSums(speech_data3),decreasing = TRUE) #sort out the words in decreasing order according to frequency.

wordcloud(words = names(word_freq3),freq = word_freq3,min.freq = 1,
          random.order = F,colors = rainbow(5))

#In the wordcloud the most common words are infrastructure,neighbour,dreath,food,kashmir
#reliant,development,freedom and much more.

word_freq4 <- data.frame(rownames(speech_matrix),rowSums(speech_matrix))
colnames(word_freq4) <- c("Word","Frequency")
View(word_freq4)

wordcloud2(word_freq4,size = 2,minSize =3,
           shape = "star",color = rainbow(50))

#Try to build a matrix for combination of words and visualize it

#For 2 words :-
bigram <- NGramTokenizer(speech_data2,Weka_control(min=2,max=2)) #Create a bigram containing two words used together.
View(bigram)
bigram_data <- data.frame(table(bigram))#Transform into data frame
View(bigram_data)

bigram_2 <- bigram_data[order(bigram_data$Freq,decreasing = TRUE),]#sort it into decending order

wordcloud(bigram_2$bigram, bigram_2$Freq, scale = c(1,0.5),
          random.order = F,  min.freq =5,
          color = rainbow(10))

wordcloud2(bigram_data,size=1,minSize=5,shape="star",color = rainbow(10))

#For 3 words together
trigram <- NGramTokenizer(speech_data2,Weka_control(min=3,max=3))
trigram_data <- data.frame(table(trigram))

wordcloud(trigram_data$trigram,trigram_data$Freq,scale = c(1,0.5),random.color = F,
          min.freq = 1,color = rainbow(30))

wordcloud2(trigram_data,size=1,minSize=5,shape="star",color = rainbow(10))


   ############### Sentiment Analysis on the speech ###############

#Here we try to extract the seniment from the speech and analyze it.
#there are so many sentiment like anger,disgust,joy,fear and much more, we try
# to divide our speech data into these category .

library(syuzhet)
library(lubridate)
library(scales)
library(reshape2)
library(dplyr)


sentment <- get_nrc_sentiment(speech_data)
head(sentment)

barplot(colSums(sentment),las=2,col = rainbow(10),ylab = 'count')
#From the plot we can say that the positive words use most of the time in speech and 
#disgust type words are less used.we can further go deep into these by analyzing that
# what are the most used positive and negative words.

sent <-  get_sentences(speech_data)#Get all the speeches into the sent as a character type
View(sent) #we got 635 sentences
class(sent) #character type
str(sent)
sent[2]
#Analyzing the sentiment by bing method
#In this method we analyze each sentence and score it the bing lexicon transform
#the sentence into binary and score it,for negative values less than zero and for 
#positive value greater than 0 assigned.

sentiment_vector <- get_sentiment(sent, method = "bing")
sentiment_vector
range(sentiment_vector)
View(sentiment_vector)
head(sentiment_vector)
#Visualization of all the sentences and there scores.

plot(sentiment_vector, type = "l", main = "Plot Trajectory",
     xlab = " Sentences ", ylab = "Emotion scores")
abline(h = 0, col = "red")


#From the top 5 sentence only fourth sentence have a score of 3 it means the sentence is
#highly positive, and all other are 0,zero means neutral.
#so from here we got the score for each sentences, now we can separately produce two
# set one from positive words and another for negative words ,and we can see from 
#wordplot the frequency of the positive and negative words.

                  #For positive sentences

positive_vector <- which(sentiment_vector>0)
head(positive_vector)#these are the sentence number that contains the positive sentence

positive_sentence <- sent[positive_vector] #collection of all positive sentence.
View(positive_sentence)
class(positive_sentence)

positive_sentence2 <- Corpus(VectorSource(positive_sentence)) #Transform into corpus

positive_tdm <- TermDocumentMatrix(positive_sentence2)#create a term doccument matrix
positive_tdm

positive_matrix <- as.matrix(positive_tdm)# Transform into matrix
head(positive_matrix)

positive_data <- data.frame(rownames(positive_matrix),rowSums(positive_matrix))#create a data set containing words and its frequency.
colnames(positive_data) <- c("Word","Frequency")

wordcloud(words = positive_data$Word,freq = positive_data$Frequency,random.order = F,
          colors = rainbow(10),min.freq = 1)

wordcloud2(positive_data,size = 2,minSize = 5,shape = "star",color = rainbow(7))

          #For Negative sentences

Negative_vector <- which(sentiment_vector<0)
head(Negative_vector)#these are the sentence number that contains the positive sentence

Negative_sentence <- sent[Negative_vector] #collection of all positive sentence.
View(Negative_sentence)
class(Negative_sentence)

Negative_sentence2 <- Corpus(VectorSource(Negative_sentence)) #Transform into corpus

Negative_tdm <- TermDocumentMatrix(Negative_sentence2)#create a term doccument matrix
Negative_tdm

Negative_matrix <- as.matrix(Negative_tdm)# Transform into matrix
head(Negative_matrix)

Negative_data <- data.frame(rownames(Negative_matrix),rowSums(Negative_matrix))#create a data set containing words and its frequency.
colnames(Negative_data) <- c("Word","Frequency")

wordcloud(words = Negative_data$Word,freq = Negative_data$Frequency,random.order = F,
          colors = rainbow(10),min.freq = 2)

wordcloud2(Negative_data,size = 2,minSize = 5,shape = "star",color = rainbow(7))

#We can also fet the most positive and negative sentence
# To extract the sentence with the most negative emotional valence
negative <- sent[which.min(sentiment_vector)]
negative

# and to extract the most positive sentence
positive <- sent[which.max(sentiment_vector)]
positive

#### Afinn method
#in this method the word is rated between -5 and 5
afinn_s_v <- get_sentiment(sent, method = "afinn")
View(afinn_s_v)
head(afinn_s_v)
range(afinn_s_v)


