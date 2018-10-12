import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import keras.backend as K
from keras.models import Model 
from keras.initializers import RandomUniform
from keras.layers import TimeDistributed, Dense, Embedding, Input, Dropout, LSTM, Bidirectional, concatenate
import numpy as np 
import tensorflow as tf
import argparse
from keras.utils import Progbar
from keras.preprocessing.sequence import pad_sequences
from keras_contrib.layers import CRF 
from string import punctuation
from keras.models import model_from_json

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required = True, choices = ('english','dutch'), help = "type of language of corpus")
ap.add_argument("-c", "--addcharinfo", choices = ('yes', 'no'), default = "yes", help = "whether using additional character information or not") 
args = vars(ap.parse_args())

if (args["dataset"] == 'english'):
	train_filename = '/storage/anhct/ner_project/eng_data/eng.train.txt'
	dev_filename = '/storage/anhct/ner_project/eng_data/eng.dev.txt'
	test_filename = '/storage/anhct/ner_project/eng_data/eng.test.txt'
	pretrained_words_filename = '/storage/anhct/ner_project/glove.6B.100d.txt'
	MAX_WORD_LEN = 52
	unique_character_set = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|"

elif (args["dataset"] == 'dutch'):
	train_filename = '/storage/anhct/ner_project/dutch_data/dut.train.txt'
	dev_filename = '/storage/anhct/ner_project/dutch_data/dut.dev.txt'
	test_filename = '/storage/anhct/ner_project/dutch_data/dut.test.txt'
	pretrained_words_filename = '/storage/anhct/ner_project/dut.100d.pretrained.txt'
	MAX_WORD_LEN = 50
	unique_character_set = " 0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZÅÖàáâãäçèéêëíîïñóôöúü.,-_()[]{}!?:;#'\"/\\%$`&=*+@^~|©°"


def load_file(filename):
	with open(filename) as file:
		sentences = []
		sentence = []
		for line in file:
			if (len(line) == 0 or line.startswith('-DOCSTART-') or line[0] == '\n'):
				if (len(sentence) > 0):
					sentences.append(sentence)
					sentence = []
				continue
			t = line.split(' ')
			h = t[-1].split()
			if (len(h)>0):
				sentence.append([t[0],h[0]])
			else:	
				sentence.append([t[0],t[-1]])

		if (len(sentence) > 0):
			sentences.append(sentence)
			sentence = []

	return sentences

def get_additional_word_info(word, add_word_info_table):
	add_word_info = 'other'

	num_of_digits = 0
	for c in word:
		if (c.isdigit()):
			num_of_digits += 1

	digit_ratio = num_of_digits / float(len(word))

	if (word.isdigit()):
		add_word_info = 'numeric'
	elif (digit_ratio > 0.5):
		add_word_info = 'mostly_numeric'
	elif (word.islower()):
		add_word_info = 'lowercase'
	elif (word.isupper()):
		add_word_info = 'uppercase'
	elif (word[0].isupper()):
		add_word_info = 'capitalization'
	elif (num_of_digits > 0):
		add_word_info = 'containing_digits'

	return add_word_info_table[add_word_info]

def get_additional_char_info(char, add_char_info_table):
	add_char_info = 'other'

	if (char.isdigit()):
		add_char_info = 'digit'
	elif (char.islower()):
		add_char_info = 'lowercase'
	elif (char.isupper()):
		add_char_info = 'uppercase'
	elif char in punctuation:
		add_char_info = 'punctuation'

	return add_char_info_table[add_char_info]

def get_batches(data):
	t = []
	for s in data:
		t.append(len(s[0]))

	t = set(t)

	batches = []
	batch_len = []

	l = 0
	for i in t:
		for s in data:
			if (len(s[0]) == i):
				batches.append(s)
				l = l + 1

		batch_len.append(l)

	return batches, batch_len

if (args["addcharinfo"] == 'yes'):
	def get_dataset(sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int, addcharinfo_to_int):
		unknown_id = word_to_int['UNKNOWN_TOKEN']
		padding_id = word_to_int['PADDING_TOKEN']

		dataset = []

		count_word = 0
		count_unknown_word = 0

		for sentence in sentences:
			word_id_set = []
			label_id_set = []
			addwordinfo_id_set = []
			char_id_set = []
			addcharinfo_id_set = []

			for word, char, label in sentence:
				count_word += 1
				if word in word_to_int:
					word_id  = word_to_int[word]
				elif word.lower() in word_to_int:
					word_id = word_to_int[word.lower()]
				else:
					word_id = unknown_id
					count_unknown_word += 1

				char_id = []
				for c in char:
					char_id.append(char_to_int[c])

				addcharinfo_id = []
				for c in char:
					addcharinfo_id.append(get_additional_char_info(c, addcharinfo_to_int))

				word_id_set.append(word_id)
				label_id_set.append(label_to_int[label])
				addwordinfo_id_set.append(get_additional_word_info(word, addwordinfo_to_int))
				char_id_set.append(char_id)
				addcharinfo_id_set.append(addcharinfo_id)

			dataset.append([word_id_set, addwordinfo_id_set, char_id_set, addcharinfo_id_set, label_id_set])

		return dataset

	def get_minibatches(dataset, batch_len):
		i = 0

		for j in batch_len:
			word_set = []
			addwordinfo_set = []
			char_set = []
			addcharinfo_set = []
			label_set = []
			data = dataset[i:j]
			i = j

			for sentence in data:
				word, addwordinfo, char, addcharinfo, label = sentence

				label = np.expand_dims(label,-1)
				word_set.append(word)
				addwordinfo_set.append(addwordinfo)
				char_set.append(char)
				addcharinfo_set.append(addcharinfo)
				label_set.append(label)

			yield np.asarray(word_set), np.asarray(addwordinfo_set), np.asarray(char_set), np.asarray(addcharinfo_set), np.asarray(label_set)

else:
	def get_dataset(sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int):
		unknown_id = word_to_int['UNKNOWN_TOKEN']
		padding_id = word_to_int['PADDING_TOKEN']

		dataset = []

		count_word = 0
		count_unknown_word = 0

		for sentence in sentences:
			word_id_set = []
			label_id_set = []
			addwordinfo_id_set = []
			char_id_set = []

			for word, char, label in sentence:
				count_word += 1
				if word in word_to_int:
					word_id  = word_to_int[word]
				elif word.lower() in word_to_int:
					word_id = word_to_int[word.lower()]
				else:
					word_id = unknown_id
					count_unknown_word += 1

				char_id = []
				for c in char:
					char_id.append(char_to_int[c])

				word_id_set.append(word_id)
				label_id_set.append(label_to_int[label])
				addwordinfo_id_set.append(get_additional_word_info(word, addwordinfo_to_int))
				char_id_set.append(char_id)

			dataset.append([word_id_set, addwordinfo_id_set, char_id_set, label_id_set])

		return dataset

	def get_minibatches(dataset, batch_len):
		i = 0

		for j in batch_len:
			word_set = []
			addwordinfo_set = []
			char_set = []
			label_set = []
			data = dataset[i:j]
			i = j

			for sentence in data:
				word, addwordinfo, char, label = sentence

				label = np.expand_dims(label,-1)
				word_set.append(word)
				addwordinfo_set.append(addwordinfo)
				char_set.append(char)
				label_set.append(label)

			yield np.asarray(word_set), np.asarray(addwordinfo_set), np.asarray(char_set), np.asarray(label_set)


def get_sentence_with_char(sentences):
	for i, sentence in enumerate(sentences):
		for j, data in enumerate(sentence):
			chars = [c for c in data[0]]
			sentences[i][j] = [data[0], chars, data[1]]

	return sentences

def get_padding(sentences, max_word_len):
	for i, sentence in enumerate(sentences):
		sentences[i][2] = pad_sequences(sentences[i][2], max_word_len, padding = 'post')
		if (args["addcharinfo"] == 'yes'):
			sentences[i][3] = pad_sequences(sentences[i][3], max_word_len, padding = 'post')

	return sentences

if (args["addcharinfo"] == 'yes'):
	def get_dataset_tag(dataset):
		correct_labels = []
		pred_labels = []
		t = Progbar(len(dataset))

		for i, data in enumerate(dataset):
			words, add_word_infos, chars, add_char_infos, labels = data
			words = np.asarray([words])
			add_word_infos = np.asarray([add_word_infos])
			chars = np.asarray([chars])
			add_char_infos = np.asarray([add_char_infos])
			pred = model.predict([words, add_word_infos, chars, add_char_infos], verbose = False)[0]
			pred = pred.argmax(axis = -1)

			correct_labels.append(labels)
			pred_labels.append(pred)

			new_i = i + 1
			t.update(new_i)

		return pred_labels, correct_labels

else:
	def get_dataset_tag(dataset):
		correct_labels = []
		pred_labels = []
		t = Progbar(len(dataset))

		for i, data in enumerate(dataset):
			words, add_word_infos, chars, labels = data
			words = np.asarray([words])
			add_word_infos = np.asarray([add_word_infos])
			chars = np.asarray([chars])
			pred = model.predict([words, add_word_infos, chars], verbose = False)[0]
			pred = pred.argmax(axis = -1)

			correct_labels.append(labels)
			pred_labels.append(pred)

			new_i = i + 1
			t.update(new_i)

		return pred_labels, correct_labels


def get_f1_score(predicted_label, correct_label, int_to_label): 
    label_pred = []    
    for sentence in predicted_label:
        label_pred.append([int_to_label[l] for l in sentence])
        
    label_correct = []    
    for sentence in correct_label:
        label_correct.append([int_to_label[l] for l in sentence])
    
    precision = get_precision_and_recall(label_pred, label_correct)
    recall = get_precision_and_recall(label_correct, label_pred)
    
    f1_score = 0
    if ((precision + recall) > 0):
        f1_score = (2.0 * precision * recall) / (precision + recall)
        
    return precision, recall, f1_score

def get_precision_and_recall(predicted_label, correct_label):
    assert(len(predicted_label) == len(correct_label))
    num_of_correct = 0
    count = 0
    
    for i in range(len(predicted_label)):
        predicted = predicted_label[i]
        correct = correct_label[i]
        assert(len(predicted) == len(correct))
        start = 0
        while (start < len(predicted)):
            if (predicted[start][0] == 'B'): 
                count += 1
                
                if (predicted[start] == correct[start]):
                    start += 1
                    check_correct = True
                    
                    while (start < len(predicted) and predicted[start][0] == 'I'): 
                        if (predicted[start] != correct[start]):
                            check_correct = False
                        
                        start += 1
                    
                    if (start < len(predicted)):
                        if (correct[start][0] == 'I'): 
                            check_correct = False
                        
                    
                    if check_correct:
                        num_of_correct += 1
                else:
                    start += 1
            else:  
                start += 1
    
    metrics = 0
    if (count > 0):    
        metrics = float(num_of_correct) / count

    return metrics

def get_precision_and_recall_label(predicted_label, correct_label, label_type):
	assert(len(predicted_label) == len(correct_label))
	num_of_correct = 0
	count = 0

	for i in range(len(predicted_label)):
		predicted = predicted_label[i]
		correct = correct_label[i]
		assert(len(predicted) == len(correct))

		start = 0
		while (start < len(predicted)):
			if (predicted[start][0] == 'B' and predicted[start] == label_type):
				count += 1

				if (predicted[start] == correct[start]):
					start += 1
					check_correct = True

					while (start < len(predicted) and predicted[start][0] == 'I'):
						if (predicted[start] != correct[start]):
							check_correct = False

						start += 1

					if (start < len(predicted)):
						if (correct[start][0] == 'I'):
							check_correct = False

					if check_correct:
						num_of_correct += 1

				else:
					start += 1

			else:
				start += 1

	metrics = 0
	if (count > 0):
		metrics = float(num_of_correct) / count

	return metrics

def get_f1_score_label(predicted_label, correct_label, int_to_label, label_type): 
    label_pred = []    
    for sentence in predicted_label:
        label_pred.append([int_to_label[l] for l in sentence])
        
    label_correct = []    
    for sentence in correct_label:
        label_correct.append([int_to_label[l] for l in sentence])
    
    precision = get_precision_and_recall_label(label_pred, label_correct, label_type)
    recall = get_precision_and_recall_label(label_correct, label_pred, label_type)
    
    f1_score = 0
    if ((precision + recall) > 0):
        f1_score = (2.0 * precision * recall) / (precision + recall)
        
    return precision, recall, f1_score


train_sentences = load_file(train_filename)
test_sentences = load_file(test_filename)
dev_sentences = load_file(dev_filename)

train_sentences = get_sentence_with_char(train_sentences)
test_sentences = get_sentence_with_char(test_sentences)
dev_sentences = get_sentence_with_char(dev_sentences)

sentence_label_set = set()
sentence_word_set = {}

for dataset in [train_sentences, dev_sentences, test_sentences]:
	for sentence in dataset:
		for word, char, label in sentence:
			sentence_label_set.add(label)
			sentence_word_set[word.lower()] = True

label_to_int = {}
for label in sentence_label_set:
	label_to_int[label] = len(label_to_int)

addwordinfo_to_int = {'numeric': 0, 'lowercase': 1, 'uppercase': 2, 'capitalization': 3, 'other': 4, 'mostly_numeric': 5, 'containing_digits': 6, 'PADDING_TOKEN': 7}
addwordinfo_embeddings = np.identity(len(addwordinfo_to_int), dtype = 'float32')

if (args["addcharinfo"] == 'yes'):
	addcharinfo_to_int = {'PADDING': 0, 'lowercase': 1, 'uppercase': 2, 'digit': 3, 'punctuation': 4, 'other': 5}
	addcharinfo_embeddings = np.identity(len(addcharinfo_to_int), dtype = 'float32')

word_to_int = {}
word_embeddings = []

pretrained_word_embeds = open(pretrained_words_filename, encoding = "utf-8")

for line in pretrained_word_embeds:
	temp = line.strip().split(" ")
	word = temp[0]

	if (len(word_to_int) == 0):
		word_to_int['PADDING_TOKEN'] = len(word_to_int)
		v = np.zeros(len(temp) - 1)
		word_embeddings.append(v)

		word_to_int['UNKNOWN_TOKEN'] = len(word_to_int)
		v = np.random.uniform(-0.25, 0.25, len(temp)-1)
		word_embeddings.append(v)

	if temp[0].lower() in sentence_word_set:
		v = np.array([float(a) for a in temp[1:]])
		word_embeddings.append(v)
		word_to_int[temp[0]] = len(word_to_int)

word_embeddings  = np.array(word_embeddings)

char_to_int = {"PADDING": 0, "UNKNOWN": 1}
for c in unique_character_set:
	char_to_int[c] = len(char_to_int)

if (args["addcharinfo"] == 'yes'):
	train_data = get_padding(get_dataset(train_sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int, addcharinfo_to_int), MAX_WORD_LEN)
	test_data = get_padding(get_dataset(test_sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int, addcharinfo_to_int), MAX_WORD_LEN)
	dev_data = get_padding(get_dataset(dev_sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int, addcharinfo_to_int), MAX_WORD_LEN)
else: 
	train_data = get_padding(get_dataset(train_sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int), MAX_WORD_LEN)
	test_data = get_padding(get_dataset(test_sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int), MAX_WORD_LEN)
	dev_data = get_padding(get_dataset(dev_sentences, word_to_int, label_to_int, addwordinfo_to_int, char_to_int), MAX_WORD_LEN)

int_to_label = {i: l for l, i in label_to_int.items()}

train_batch, train_batch_len = get_batches(train_data)
test_batch, test_batch_len = get_batches(test_data)
dev_batch, dev_batch_len = get_batches(dev_data)


word_input = Input(shape = (None,), dtype = 'int32', name = 'word_input')
embed_word = Embedding(input_dim = word_embeddings.shape[0], output_dim = word_embeddings.shape[1], weights = [word_embeddings], trainable = False)(word_input)

addwordinfo_input = Input(shape = (None,), dtype = 'int32', name = 'addwordinfo_input')
embed_addwordinfo = Embedding(input_dim = addwordinfo_embeddings.shape[0], output_dim = addwordinfo_embeddings.shape[1], weights = [addwordinfo_embeddings], trainable = False)(addwordinfo_input)

char_input = Input(shape = (None, MAX_WORD_LEN, ), name = 'char_input')
embed_char = TimeDistributed(Embedding(len(char_to_int), 30, embeddings_initializer = RandomUniform(minval = -0.5, maxval = 0.5), mask_zero=True), name = 'char_embedding')(char_input)
embed_char_out = Dropout(0.5)(embed_char)

if (args["addcharinfo"] == 'yes'):
	addcharinfo_input = Input(shape = (None, MAX_WORD_LEN, ), name = 'addcharinfo_input')
	embed_addcharinfo = Embedding(input_dim = addcharinfo_embeddings.shape[0], output_dim = addcharinfo_embeddings.shape[1], weights = [addcharinfo_embeddings], trainable = False)(addcharinfo_input)

	embed_char_new = concatenate([embed_char_out, embed_addcharinfo])
else:
	embed_char_new = embed_char_out

char_representation = TimeDistributed(Bidirectional(LSTM(20)))(embed_char_new)

char_representation = Dropout(0.5)(char_representation)

word_representation = concatenate([embed_word, embed_addwordinfo, char_representation])

output = Bidirectional(LSTM(200, return_sequences=True, dropout=0.50, recurrent_dropout=0.25))(word_representation)

output = TimeDistributed(Dense(100, activation = 'tanh'))(output)

crf = CRF(len(label_to_int), sparse_target=True, learn_mode = 'join', test_mode = 'viterbi')
output = crf(output)

if (args["addcharinfo"] == 'yes'):
	model = Model(inputs=[word_input, addwordinfo_input, char_input, addcharinfo_input], outputs=[output])
	model.compile(loss=crf.loss_function, optimizer='nadam')
	model.summary()
else:
	model = Model(inputs=[word_input, addwordinfo_input, char_input], outputs=[output])
	model.compile(loss=crf.loss_function, optimizer='nadam')
	model.summary()

train_loss_list = []
dev_loss_list = []
train_f1_score_list = []
dev_f1_score_list = []

if (args["dataset"] == "english"):
	epsilon = 0.0005
elif (args["dataset"] == "dutch"):
	epsilon = 0.0002

mod_epsilon = 0.005
epochs = 150

if (args["addcharinfo"] == 'yes'):
	for epoch in range(epochs):    
		train_loss = 0
		dev_loss = 0
		print("\nEpoch %d/%d" % (epoch + 1,epochs))
		a = Progbar(len(train_batch_len))
		for i, batch in enumerate(get_minibatches(train_batch,train_batch_len)):
			w, awi, c, aci, l = batch      
			train_loss += model.train_on_batch([w, awi, c, aci], l)
			newi = i + 1
			a.update(newi)

		print(' ')

		for i, batch in enumerate(get_minibatches(dev_batch, dev_batch_len)):
			w, awi, c, aci, l = batch
			dev_loss += model.test_on_batch([w, awi, c, aci], l)

		train_loss = train_loss / len(train_batch_len)
		dev_loss = dev_loss / len(dev_batch_len)

		print("- Training Loss: %.5f - Dev Loss: %.5f" % (train_loss, dev_loss))

		trainpred, traincorrect = get_dataset_tag(train_batch)
		pre_train, rec_train, f1_train = get_f1_score(trainpred, traincorrect, int_to_label)

		devpred, devcorrect = get_dataset_tag(dev_batch)
		pre_dev, rec_dev, f1_dev = get_f1_score(devpred, devcorrect, int_to_label)

		print(" - Train F1 Score: %.4f - Dev F1 Score: %.4f" % (f1_train, f1_dev))

		train_loss_list.append(train_loss)
		dev_loss_list.append(dev_loss)

		train_f1_score_list.append(f1_train)
		dev_f1_score_list.append(f1_dev)

		if (len(dev_loss_list) > 2):
			if (abs(dev_loss_list[epoch] - dev_loss_list[epoch-1]) < epsilon and abs(dev_loss_list[epoch-1] - dev_loss_list[epoch-2]) < epsilon): 
				break
			if (args["dataset"] == "english"):
				if (epoch >= 100):
					if (abs(dev_loss_list[epoch] - dev_loss_list[epoch-1]) < mod_epsilon and abs(dev_loss_list[epoch-1] - dev_loss_list[epoch-2]) < mod_epsilon):
						break
			elif (args["dataset"] == "dutch"):
				if (epoch >= 75):
					if (abs(dev_loss_list[epoch] - dev_loss_list[epoch-1]) < mod_epsilon and abs(dev_loss_list[epoch-1] - dev_loss_list[epoch-2]) < mod_epsilon):
						break

else:
	for epoch in range(epochs):    
		train_loss = 0
		dev_loss = 0
		print("\nEpoch %d/%d" % (epoch + 1,epochs))
		a = Progbar(len(train_batch_len))
		for i, batch in enumerate(get_minibatches(train_batch,train_batch_len)):
			w, awi, c, l = batch      
			train_loss += model.train_on_batch([w, awi, c], l)
			newi = i + 1
			a.update(newi)

		print(' ')

		for i, batch in enumerate(get_minibatches(dev_batch, dev_batch_len)):
			w, awi, c, l = batch
			dev_loss += model.test_on_batch([w, awi, c], l)

		train_loss = train_loss / len(train_batch_len)
		dev_loss = dev_loss / len(dev_batch_len)

		print("- Training Loss: %.5f - Dev Loss: %.5f" % (train_loss, dev_loss))

		trainpred, traincorrect = get_dataset_tag(train_batch)
		pre_train, rec_train, f1_train = get_f1_score(trainpred, traincorrect, int_to_label)

		devpred, devcorrect = get_dataset_tag(dev_batch)
		pre_dev, rec_dev, f1_dev = get_f1_score(devpred, devcorrect, int_to_label)

		print(" - Train F1 Score: %.4f - Dev F1 Score: %.4f" % (f1_train, f1_dev))

		train_loss_list.append(train_loss)
		dev_loss_list.append(dev_loss)

		train_f1_score_list.append(f1_train)
		dev_f1_score_list.append(f1_dev)

		if (len(dev_loss_list) > 2):
			if (abs(dev_loss_list[epoch] - dev_loss_list[epoch-1]) < epsilon and abs(dev_loss_list[epoch-1] - dev_loss_list[epoch-2]) < epsilon): 
				break
			if (args["dataset"] == "english"):
				if (epoch >= 100):
					if (abs(dev_loss_list[epoch] - dev_loss_list[epoch-1]) < mod_epsilon and abs(dev_loss_list[epoch-1] - dev_loss_list[epoch-2]) < mod_epsilon):
						break
			elif (args["dataset"] == "dutch"):
				if (epoch >= 75):
					if (abs(dev_loss_list[epoch] - dev_loss_list[epoch-1]) < mod_epsilon and abs(dev_loss_list[epoch-1] - dev_loss_list[epoch-2]) < mod_epsilon):
						break

    
predicted_labels, correct_labels = get_dataset_tag(dev_batch)        
pre_dev, rec_dev, f1_dev = get_f1_score(predicted_labels, correct_labels, int_to_label)
print(" Dev-Data: Precision: %.4f, Recall: %.4f, F1-score: %.4f" % (pre_dev, rec_dev, f1_dev))

predicted_labels, correct_labels = get_dataset_tag(test_batch)        
pre_test, rec_test, f1_test= get_f1_score(predicted_labels, correct_labels, int_to_label)
print(" Test-Data: Precision: %.4f, Recall: %.4f, F1-score: %.4f" % (pre_test, rec_test, f1_test))

model_json = model.to_json()
with open("BLSTM_CRF_model_v0.json", "w") as json_file:
    json_file.write(model_json)

model.save_weights("BLSTM_CRF_weights_v0.h5")
print('#Finish save model v0.')

label_type_set = ['B-PER', 'B-LOC', 'B-ORG', 'B-MISC']
for label in label_type_set:
	precision_test, recall_test, f1_score_test = get_f1_score_label(predicted_labels, correct_labels, int_to_label, label)
	print("\n%s : [Precision: %.4f] - [Recall: %.4f] - [F1-score: %.4f]" % (label[2:], precision_test, recall_test, f1_score_test))


with open('BLSTM_CRF_loss_v0.txt', 'w') as f:
	f.writelines("%f\n" % i for i in train_loss_list)
	f.writelines("\n")
	f.writelines("%f\n" % i for i in dev_loss_list)

with open('BLSTM_CRF_f1score_v0.txt', 'w') as f:
	f.writelines("%f\n" % i for i in train_f1_score_list)
	f.writelines("\n")
	f.writelines("%f\n" % i for i in dev_f1_score_list)
