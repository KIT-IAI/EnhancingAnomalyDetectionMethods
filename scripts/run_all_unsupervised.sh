#!/bin/bash
cd ../detection_pipelines
python run_unsupervised_methods.py --anomalies 10 --base iForest --generator-method cvae cinn --contaminations 0.95  --type unusual
python run_unsupervised_methods.py --anomalies 30 --base iForest --generator-method cvae cinn --contaminations 0.85 --type unusual
python run_unsupervised_methods.py --anomalies 40 --base iForest --generator-method cvae cinn --contaminations 0.8 --type unusual
python run_unsupervised_methods.py --anomalies 50 --base iForest --generator-method cvae cinn --contaminations 0.75 --type unusual
python run_unsupervised_methods.py --anomalies 10 --base LOF --generator-method cvae cinn --contaminations 0.95 --type unusual
python run_unsupervised_methods.py --anomalies 30 --base LOF --generator-method cvae cinn --contaminations 0.85 --type unusual
python run_unsupervised_methods.py --anomalies 40 --base LOF --generator-method cvae cinn --contaminations 0.8 --type unusual
python run_unsupervised_methods.py --anomalies 50 --base LOF --generator-method cvae cinn --contaminations 0.75 --type unusual
python run_unsupervised_methods.py --anomalies 10 --base AE --generator-method cvae cinn --contaminations 0.95 --type unusual
python run_unsupervised_methods.py --anomalies 30 --base AE --generator-method cvae cinn --contaminations 0.85 --type unusual
python run_unsupervised_methods.py --anomalies 40 --base AE --generator-method cvae cinn --contaminations 0.8 --type unusual
python run_unsupervised_methods.py --anomalies 50 --base AE --generator-method cvae cinn --contaminations 0.75 --type unusual
python run_unsupervised_methods.py --anomalies 10 --base VAE --generator-method cvae cinn --contaminations 0.95 --type unusual
python run_unsupervised_methods.py --anomalies 30 --base VAE --generator-method cvae cinn --contaminations 0.85 --type unusual
python run_unsupervised_methods.py --anomalies 40 --base VAE --generator-method cvae cinn --contaminations 0.8 --type unusual
python run_unsupervised_methods.py --anomalies 50 --base VAE --generator-method cvae cinn --contaminations 0.75 --type unusual
