profile=mlops

.PHONY: init-env
init-env:
	@conda env create -f project-dependencies.yml

.PHONY: run-training
run-training: 
	@python model_training.py

.PHONY: run-service
run-service:
	@streamlit run src/app.py

.PHONY: aws-login
aws-login:
	@aws sso login --sso-session ${profile}

.PHONY: aws-deploy
aws-deploy:
	@eb create --timeout 25 --instance-types "t3.small"  sound-classification-env

.PHONY: aws-delete-env
	@eb terminate sound-classification-env