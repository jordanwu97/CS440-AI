temp = np.zeros((self.feature_dim,self.num_value,self.num_class))
for example_idx in range(train_set.shape[0]):
	label = train_label[example_idx]
	image = train_set[example_idx]
	for pixel_idx in range(train_set.shape[1]):
		temp[pixel_idx][image[pixel_idx]][label] += 1

print (np.all(np.equal(self.likelihood, temp)))