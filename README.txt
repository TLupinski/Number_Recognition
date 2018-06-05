Liste des fichiers importants :
	train_custom.py:
		- Fichier principale pour l'apprentissage.
		Il faut définir manuellement le modèle à charger, le dossier contenant la dataset à utiliser, le dossier de sortie où seront sauvegardés les poids et les valeurs au cours de l'apprentissage, la taille des images, le nombre d'images à utiliser et la taille maximale des chaînes en sorties. Tout ceci ce fait soit juste après les import pour les noms de dossiers, soit au début de train pour les paramètres entier.
	test_custom.py:
		- Fichier principale pour l'évaluation.
		Il faut définir manuellement le modèle à charger, le dossier contenant la dataset à utiliser, le dossier de sortie où sont sauvegardés les poids à utiliser, la taille des images, le nombre d'images à utiliser et la taille maximale des chaînes en sorties.
	custom_model.py:
		- Tous les modèles disponibles sont définis dans ce fichier.
	network_helper.py :
		- Définition de TextImageGenerator : Générateur de données utilisé pendant les apprentissages/évaluations par train_custom/test_custom. C'est ici qu'il faut définir la façon de charger les données dans get_train_images()/get_val_images() et get_train_text()/get_val_text().
	edistance.py :


Credit to farizrahman4u for libraries:
seq2seq		https://github.com/farizrahman4u/seq2seq/
recurrentshop	https://github.com/farizrahman4u/recurrentshop/
Credit to Ben Lambert for edit distance :
edistance.py	https://github.com/belambert/edit-distance
