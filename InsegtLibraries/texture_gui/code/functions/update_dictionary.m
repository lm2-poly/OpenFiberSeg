function dictionary = update_dictionary(dictionary,gui_dictprob)

M = dictionary.options.patch_size;
nr_dict_patches = size(dictionary.tree,2);
L = size(gui_dictprob,2); % nr labels
gui_dictprob = reshape(gui_dictprob,[M^2,nr_dict_patches,L]);
gui_dictprob = permute(gui_dictprob,[1,3,2]);
gui_dictprob = reshape(gui_dictprob,[M^2*L,nr_dict_patches]);
dictionary.dictprob = gui_dictprob;