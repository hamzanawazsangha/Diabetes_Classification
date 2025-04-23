# Importing necessary libraries
import pickle
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

# Streamlit page configuration
st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 5px;
            padding: 0.5rem 1rem;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
        .stSelectbox, .stNumberInput {
            margin-bottom: 1rem;
        }
        .stAlert {
            border-radius: 5px;
        }
        .stContainer {
            background-color: white;
            padding: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        .header {
            color: #2c3e50;
        }
        .feature-card {
            padding: 1rem;
            background-color: #e8f4f8;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Streamlit Containers
header = st.container()
features = st.container()
input_section = st.container()
prediction_section = st.container()
preprocessing_section = st.container()

# Header Section
with header:
    st.title('🩺 Diabetes Prediction System')
    st.markdown("""
        <div style='text-align: justify;'>
        Diabetes has become a global health issue with increasing prevalence rates. 
        Early detection can significantly improve treatment outcomes and quality of life for patients. 
        This predictive system leverages health parameters to predict the likelihood of diabetes in a person.
        </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

# Features Section
with features:
    st.header('📋 Features Required for Prediction')
    st.write('The following features are required for accurate prediction:')
    
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🔍 Demographic Information"):
            st.markdown("""
            - **Age**: Age of the patient (years)
            - **BMI**: Body Mass Index (kg/m²)
            """)
            
    with col2:
        with st.expander("🧪 Medical Indicators"):
            st.markdown("""
            - **HbA1c Level**: Level of Hemoglobin A1c in blood (%)
            - **Blood Glucose Level**: Glucose level in blood (mg/dL)
            """)
    
    with st.expander("💓 Medical History"):
        st.markdown("""
        - **Hypertension**: Whether the patient has hypertension (Yes/No)
        - **Heart Disease**: Whether the patient has heart disease (Yes/No)
        """)
    
    st.markdown("---")

# Input Section
with input_section:
    st.header('📊 Prediction Machine')
    
    with st.form(key='prediction_form'):
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Patient Information")
            age = st.number_input('Age', min_value=0, max_value=120, value=30, step=1)
            bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
            hb = st.number_input('HbA1c Level', min_value=3.0, max_value=20.0, value=5.7, step=0.1,
                                help="Normal range: 4-5.6%, Prediabetes: 5.7-6.4%, Diabetes: 6.5% or higher")
            
        with col2:
            st.subheader("Medical History")
            gbl = st.number_input('Blood Glucose Level', min_value=50, max_value=300, value=100, step=1,
                                help="Normal fasting level: 70-100 mg/dL")
            hyp = st.selectbox('Hypertension', options=['No', 'Yes'], 
                             help="High blood pressure condition")
            heart = st.selectbox('Heart Disease', options=['No', 'Yes'])
            
        st.subheader("Model Selection")
        models = st.selectbox('Choose Prediction Model', 
                             options=['Logistic Regression', 'Random Forest', 'Decision Tree', 
                                      'Support Vector Classifier', 'KNeighbors'],
                             index=1,
                             help="Random Forest generally provides good balance of accuracy and interpretability")
        
        submit_button = st.form_submit_button(label='🔮 Predict Diabetes Risk')

    # Preprocessing Input
    if submit_button:
        # Encode categorical inputs
        hyp = 1 if hyp == 'Yes' else 0
        heart = 1 if heart == 'Yes' else 0

        # Create a DataFrame for input data
        input_data = pd.DataFrame({
            'age': [age],
            'bmi': [bmi],
            'HbA1c_level': [hb],
            'blood_glucose_level': [gbl],
            'hypertension': [hyp],
            'heart_disease': [heart]
        })

        # Load and apply the scaler
        try:
            with open('scale.pkl', 'rb') as f:
                scaler = pickle.load(f)
            num_features = input_data[['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']]
            scaled_features = pd.DataFrame(scaler.transform(num_features), columns=num_features.columns)
            input_data = pd.concat([scaled_features, input_data[['hypertension', 'heart_disease']]], axis=1)

            # Load Models
            models_dict = {
                'Logistic Regression': 'LR.pkl',
                'Support Vector Classifier': 'svc.pkl',
                'Decision Tree': 'DT.pkl',
                'Random Forest': 'RF.pkl',
                'KNeighbors': 'Kn.pkl'
            }
            
            with open(models_dict[models], 'rb') as f:
                selected_model = pickle.load(f)

            # Make Predictions with probability
            prediction = selected_model.predict(input_data)
            prediction_proba = selected_model.predict_proba(input_data)
            
            with prediction_section:
                st.header('📌 Prediction Results')
                
                # Create a metric card
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.error('High Risk of Diabetes')
                        st.markdown(f"**Probability:** {prediction_proba[0][1]*100:.1f}%")
                        st.warning("Recommendation: Consult with a healthcare professional for further evaluation.")
                    else:
                        st.success('Low Risk of Diabetes')
                        st.markdown(f"**Probability:** {prediction_proba[0][0]*100:.1f}%")
                        st.info("Recommendation: Maintain healthy lifestyle with regular checkups.")
                
                with col2:
                    # Show feature importance if available
                    if hasattr(selected_model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'Feature': ['Age', 'BMI', 'HbA1c', 'Glucose', 'Hypertension', 'Heart Disease'],
                            'Importance': selected_model.feature_importances_
                        }).sort_values('Importance', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(8, 4))
                        sns.barplot(x='Importance', y='Feature', data=importance_df, palette='viridis', ax=ax)
                        plt.title('Feature Importance')
                        st.pyplot(fig)
                    else:
                        st.info("Feature importance not available for this model.")
                
                st.markdown("---")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.warning("Please ensure all model files are available in the correct location.")

# Preprocessing and Evaluation Section
with preprocessing_section:
    st.header('📈 Model Evaluation')
    
    tab1, tab2, tab3, tab4 = st.tabs(["Model Comparison", "Confusion Matrix", "Classification Report", "ROC Curve"])
    
    with tab1:
        st.subheader('Model Performance Comparison')
        with st.form(key='comparison_form'):
            comparison_models = st.multiselect(
                'Select models to compare:',
                ['Original', 'Logistic Regression', 'Random Forest', 'Decision Tree', 'SVC', 'KNeighbors'],
                default=['Logistic Regression', 'Random Forest']
            )
            submit_comparison = st.form_submit_button(label='Generate Comparison')

        if submit_comparison:
            try:
                with open('df5.pkl', 'rb') as f:
                    comparison_data = pickle.load(f)

                if comparison_models:
                    model_mapping = {
                        'Original': 'original',
                        'Logistic Regression': 'LR',
                        'Random Forest': 'RF',
                        'Decision Tree': 'DT',
                        'SVC': 'SVC',
                        'KNeighbors': 'KNN'
                    }
                    
                    selected_columns = [model_mapping[model] for model in comparison_models if model in model_mapping]
                    
                    # Plot KDE for selected models
                    st.subheader('Performance Distribution')
                    fig, ax = plt.subplots(figsize=(10, 6))
                    for col, label in zip(selected_columns, comparison_models):
                        comparison_data[col].plot(kind='kde', label=label, ax=ax)

                    plt.legend()
                    plt.title('Model Performance Distribution Comparison')
                    plt.xlabel('Prediction Values')
                    plt.ylabel('Density')
                    st.pyplot(fig)
                    
                    # Show accuracy comparison
                    st.subheader('Accuracy Comparison')
                    accuracy_data = []
                    for model in comparison_models:
                        if model != 'Original':
                            try:
                                with open(f"{model_mapping[model]}_s.pkl", 'rb') as f:
                                    accuracy = pickle.load(f)
                                accuracy_data.append({'Model': model, 'Accuracy': accuracy})
                            except:
                                continue
                    
                    if accuracy_data:
                        accuracy_df = pd.DataFrame(accuracy_data)
                        fig, ax = plt.subplots(figsize=(10, 4))
                        sns.barplot(x='Model', y='Accuracy', data=accuracy_df, palette='coolwarm', ax=ax)
                        plt.ylim(0, 1)
                        plt.title('Model Accuracy Comparison')
                        plt.xticks(rotation=45)
                        for p in ax.patches:
                            ax.annotate(f"{p.get_height():.2f}", 
                                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                                        ha='center', va='center', 
                                        xytext=(0, 10), 
                                        textcoords='offset points')
                        st.pyplot(fig)
                else:
                    st.warning("Please select at least one model for comparison.")
            except Exception as e:
                st.error(f"Error loading comparison data: {str(e)}")
    
    with tab2:
        st.subheader('Confusion Matrix Analysis')
        selected_model = st.selectbox(
            'Select model:',
            ['Random Forest', 'SVC', 'KNeighbors', 'Decision Tree', 'Logistic Regression'],
            key='confusion_matrix_model'
        )
        
        try:
            cm_files = {
                'Random Forest': 'cm_RF.pkl',
                'SVC': 'cm_svc.pkl',
                'KNeighbors': 'cm_Kn.pkl',
                'Decision Tree': 'cm_DT.pkl',
                'Logistic Regression': 'cm_LR.pkl'
            }
            with open(cm_files[selected_model], 'rb') as f:
                cm = pickle.load(f)
                
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', cbar=False, 
                        xticklabels=['No Diabetes', 'Diabetes'], 
                        yticklabels=['No Diabetes', 'Diabetes'],
                        ax=ax)
            plt.title(f"Confusion Matrix: {selected_model}")
            plt.xlabel('Predicted')
            plt.ylabel('Actual')
            st.pyplot(fig)
            
            # Calculate metrics from confusion matrix
            tn, fp, fn, tp = cm.ravel()
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy:.2%}")
            col2.metric("Precision", f"{precision:.2%}")
            col3.metric("Recall", f"{recall:.2%}")
            col4.metric("F1 Score", f"{f1:.2%}")
            
        except Exception as e:
            st.error(f"Error loading confusion matrix: {str(e)}")
    
    with tab3:
        st.subheader('Detailed Classification Report')
        selected_model_cls = st.selectbox(
            'Select model:',
            ['Random Forest', 'SVC', 'KNeighbors', 'Decision Tree', 'Logistic Regression'],
            key='classification_report_model'
        )
        
        try:
            cl_files = {
                'Random Forest': 'RF_class.pkl',
                'SVC': 'svc_class.pkl',
                'KNeighbors': 'Kn_class.pkl',
                'Decision Tree': 'DT_class.pkl',
                'Logistic Regression': 'LR_class.pkl'
            }

            with open(cl_files[selected_model_cls], 'rb') as f:
                classification_report_data = pickle.load(f)

            if isinstance(classification_report_data, pd.DataFrame):
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.heatmap(classification_report_data.iloc[:-1, :-1], annot=True, 
                           cmap='YlGnBu', fmt=".2f", ax=ax,
                           annot_kws={"size": 12})
                ax.set_title(f"Classification Report: {selected_model_cls}")
                st.pyplot(fig)
            else:
                st.text(f"Classification Report for {selected_model_cls}:\n")
                st.text(classification_report_data)
                
        except Exception as e:
            st.error(f"Error loading classification report: {str(e)}")
    
    with tab4:
        st.subheader('ROC Curve Analysis')
        selected_model_roc = st.selectbox(
            'Select model:', 
            ['Random Forest', 'SVC', 'KNeighbors', 'Decision Tree', 'Logistic Regression'],
            key='roc_model'
        )
        
        try:
            roc_files = {
                'Random Forest': ('RF_fpr.pkl', 'RF_tpr.pkl', 'RF_roc_auc.pkl'),
                'SVC': ('svc_fpr.pkl', 'svc_tpr.pkl', 'svc_roc_auc.pkl'),
                'KNeighbors': ('Kn_fpr.pkl', 'Kn_tpr.pkl', 'Kn_roc_auc.pkl'),
                'Decision Tree': ('DT_fpr.pkl', 'DT_tpr.pkl', 'DT_roc_auc.pkl'),
                'Logistic Regression': ('LR_fpr.pkl', 'LR_tpr.pkl', 'LR_roc_auc.pkl')
            }
            fpr_file, tpr_file, roc_auc_file = roc_files[selected_model_roc]
            
            with open(fpr_file, 'rb') as f:
                fpr = pickle.load(f)
            with open(tpr_file, 'rb') as f:
                tpr = pickle.load(f)
            with open(roc_auc_file, 'rb') as f:
                roc_auc = pickle.load(f)
                
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (AUC = {roc_auc:.2f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.title(f'ROC Curve: {selected_model_roc}')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend(loc='lower right')
            plt.grid(True)
            st.pyplot(fig)
            
            # Interpretation of AUC score
            st.markdown("""
            **AUC Score Interpretation:**
            - 0.90-1.00 = Excellent
            - 0.80-0.90 = Good
            - 0.70-0.80 = Fair
            - 0.60-0.70 = Poor
            - 0.50-0.60 = Fail
            """)
            
        except Exception as e:
            st.error(f"Error loading ROC curve data: {str(e)}")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: gray;'>
        Diabetes Prediction System © 2023 | For educational purposes only
    </div>
""", unsafe_allow_html=True)
