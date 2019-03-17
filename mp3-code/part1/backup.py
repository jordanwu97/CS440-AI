example_arange = np.zeros(train_set.shape)
pixel_arange = np.zeros(train_set.shape)

for i in range(train_set.shape[0]):
	example_arange[i,:] = i

for i in range(train_set.shape[1]):
	pixel_arange[:,i] = i

print (example_arange)

self.feature[pixel_idx][train_set[example_idx][pixel_idx]][train_label[example_idx]] += 1