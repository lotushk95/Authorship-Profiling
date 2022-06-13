# introduction
This is group G. We worked to know the author of letter is young or old people, and this is authorship profiling.
We used two methods to analyze them. Counting the number of young words and old words that we selected from google,
and calculating cosine similarity using US young friend's letter.

# explanation of program
At first, we obtained letter samples from Letters anonymous, and tokenize the sample sentences to get easier to
count the number of words that are in list we made.

# first main part... counting
We counted the number of words in original young word list and old word list. This is first main part of ths program.
It returns the number of our words list word, but now we don't have old words in sample letters.

# second main part... cosine similarity compare to US friend
Next, we tried to calculate cosine similarity using my US young friend's letter as sample_11.txt. This is second main part of this program, 
and this is interesting trial. If sample letter is similar to it, the program returns positive number.
Unfortunately, this time the program returns only positive number.

# focus on young
We don't have old words in sample letters and cosine similarity is positive, so we focused on judging young or not.

# interesting part... cosine similarity, evaluate value
This time, we tried to use cosine similarity using my US friend's letter. That was difficult, but
that may be unique idea.
And we thought out the value that is used for evaluation. The evaluation value that we used in this program 
is addition of return value of counting part and cosine similarity part. This time we obtained really small cosine similarity, so we had to make new value.
We think this is interesting method.

# how is the program work ---> good or poor
Our program works good for profiling young or not, but it works poor for profiling old or not.
cosine similarity function only returned positive values, so we can only say those letters are
written by young people. And sample letters don't have older word list word.
This is reason why our program is good for profiling young or not.

# evidence of the performance ---> our answer and program answer matches
We tried to judge young or not by ourselves, and our answers matches the program answer.
So at least, it can analyze young or not, and this is evidence.

# Discuss any issues and how the issue might be solved.
We think original word lists are weak for profiling, so we have to develop it to improve this program.
And we used only one base letter to calculate cosine similarity, that was not enough, probably.
If we use 10 letters, 20 letters or more, we can obtain more exact result.
So we can make the program better by developing each word list, and using more letters.

we tried to run this program for young text in twitter as additinal trial. and this is result.
we can say this similarity is lager than sample letters, but still small.
Perhaps, sample letters and text in twitter are too short to calculate similarity.

This trial is so interesting, and that's all our presentation.
=======
# Authorship-Profiling
the program to determine if the author of letter is young or old
>>>>>>> adb0e5765cbdee9bf1104912475d9009ef4be890
