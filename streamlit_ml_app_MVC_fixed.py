"""
🚀 IMPROVED ML FRONTEND - BETTER VISUALIZATION
==============================================
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
import time
import json
from typing import Dict, Any
import threading
import queue

# Import deiner Original-Klassen (angepasst)
import os
import logging
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score as sklearn_r2
import tensorflow as tf
import keras
from keras.optimizers import Adam
from keras.layers import Dense, Dropout
from keras.models import Sequential
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StreamlitMLTrainer:
    """🎯 ML-KLASSE FÜR STREAMLIT INTEGRATION"""

    def __init__(self):
        self.model = None
        self.scaler = None
        self.y_scaler = None
        self.history = None
        self.training_active = False

        # UI Update Queue
        if 'training_queue' not in st.session_state:
            st.session_state.training_queue = queue.Queue()
        if 'training_logs' not in st.session_state:
            st.session_state.training_logs = []

    def log_to_ui(self, message: str):
        """📝 Sende Log-Message an UI"""
        st.session_state.training_logs.append(f"{datetime.now().strftime('%H:%M:%S')} - {message}")
        if len(st.session_state.training_logs) > 50:
            st.session_state.training_logs = st.session_state.training_logs[-50:]

    def load_and_preprocess_data(self, test_size: float = 0.2, val_size: float = 0.2):
        """📊 Daten laden und vorverarbeiten"""
        self.log_to_ui("📊 Lade Diabetes-Dataset...")

        dataset = load_diabetes()
        X, y = dataset.data, dataset.target.reshape(-1, 1)

        self.log_to_ui(f"✅ Dataset geladen: {X.shape[0]} Samples, {X.shape[1]} Features")

        # Splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size, random_state=42
        )

        # Standardisierung
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train).astype(np.float32)
        X_val_scaled = self.scaler.transform(X_val).astype(np.float32)
        X_test_scaled = self.scaler.transform(X_test).astype(np.float32)

        self.y_scaler = StandardScaler()
        y_train_scaled = self.y_scaler.fit_transform(y_train).astype(np.float32)
        y_val_scaled = self.y_scaler.transform(y_val).astype(np.float32)
        y_test_scaled = self.y_scaler.transform(y_test).astype(np.float32)

        self.log_to_ui(f"🔄 Daten verarbeitet: {len(X_train)} Train, {len(X_val)} Val, {len(X_test)} Test")

        return (X_train_scaled, y_train_scaled), (X_val_scaled, y_val_scaled), (X_test_scaled, y_test_scaled)

    def build_model(self, hidden_layers: list, dropout_rate: float, l2_reg: float, learning_rate: float):
        """🏗️ Modell erstellen"""
        self.log_to_ui(f"🏗️ Erstelle Modell: {len(hidden_layers)} Hidden Layers {hidden_layers}")

        model = Sequential(name="StreamlitDiabetesRegression")

        # Hidden Layers
        for i, units in enumerate(hidden_layers):
            if i == 0:
                model.add(Dense(
                    units=units, input_shape=(10,),
                    activation='relu', kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(l2_reg)
                ))
            else:
                model.add(Dense(
                    units=units, activation='relu',
                    kernel_initializer='he_normal',
                    kernel_regularizer=keras.regularizers.l2(l2_reg)
                ))

            model.add(Dropout(dropout_rate))

        # Output Layer
        model.add(Dense(1, kernel_initializer='he_normal'))

        # Kompilieren
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='mse',
            metrics=['mae']
        )

        total_params = model.count_params()
        self.log_to_ui(f"✅ Modell erstellt: {total_params:,} Parameter")

        return model

    def train_model_with_ui_updates(self, model, train_data, val_data,
                                   max_epochs: int, batch_size: int, patience: int):
        """🚀 Training mit Live-UI-Updates"""

        X_train, y_train = train_data
        X_val, y_val = val_data

        self.log_to_ui(f"🚀 Starte Training: {max_epochs} max Epochs, Batch Size {batch_size}")

        # Custom Callback für UI-Updates
        class StreamlitCallback(keras.callbacks.Callback):
            def __init__(self, ui_logger):
                self.ui_logger = ui_logger
                self.start_time = time.time()

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % 5 == 0 or epoch < 10:
                    elapsed = time.time() - self.start_time
                    self.ui_logger(
                        f"📈 Epoch {epoch+1}: Loss={logs['loss']:.4f}, "
                        f"Val_Loss={logs['val_loss']:.4f}, Zeit={elapsed:.1f}s"
                    )

            def on_train_end(self, logs=None):
                total_time = time.time() - self.start_time
                self.ui_logger(f"✅ Training beendet nach {total_time:.1f} Sekunden")

        # Callbacks
        callbacks = [
            StreamlitCallback(self.log_to_ui),
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=patience//2, min_lr=1e-7, verbose=0)
        ]

        # Training
        self.history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=max_epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=0
        )

        return model

    def evaluate_model_with_ui(self, model, test_data):
        """📊 Evaluation mit UI-Feedback"""
        self.log_to_ui("🎯 Starte Modell-Evaluierung...")

        X_test, y_test = test_data
        y_pred_scaled = model.predict(X_test, verbose=0)

        # Zurück-transformieren
        y_test_original = self.y_scaler.inverse_transform(y_test)
        y_pred_original = self.y_scaler.inverse_transform(y_pred_scaled)

        # Metriken berechnen
        results = {
            'mse': float(mean_squared_error(y_test_original, y_pred_original)),
            'rmse': float(np.sqrt(mean_squared_error(y_test_original, y_pred_original))),
            'mae': float(mean_absolute_error(y_test_original, y_pred_original)),
            'r2': float(sklearn_r2(y_test_original, y_pred_original)),
            'mape': float(np.mean(np.abs((y_test_original - y_pred_original) / y_test_original)) * 100),
            'test_samples': len(y_test),
            'y_true': y_test_original.flatten(),
            'y_pred': y_pred_original.flatten()
        }

        self.log_to_ui(f"🎯 Evaluierung abgeschlossen: R²={results['r2']:.4f}, MSE={results['mse']:.1f}")

        return results

    def run_complete_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """🎯 Komplettes Training Pipeline"""
        try:
            st.session_state.training_logs = []
            self.training_active = True

            # 1. Daten laden
            train_data, val_data, test_data = self.load_and_preprocess_data(
                config['test_size'], config['validation_size']
            )

            # 2. Modell erstellen
            self.model = self.build_model(
                config['hidden_layers'],
                config['dropout_rate'],
                config['l2_reg'],
                config['learning_rate']
            )

            # 3. Training
            self.model = self.train_model_with_ui_updates(
                self.model, train_data, val_data,
                config['max_epochs'], config['batch_size'], config['patience']
            )

            # 4. Evaluation
            results = self.evaluate_model_with_ui(self.model, test_data)

            self.training_active = False
            self.log_to_ui("🏆 Pipeline komplett abgeschlossen!")

            return results

        except Exception as e:
            self.training_active = False
            self.log_to_ui(f"❌ Fehler: {e}")
            return {"error": str(e)}

def create_architecture_visualization(hidden_layers):
    """Erstellt eine saubere Architektur-Visualisierung"""
    fig = go.Figure()

    # Layer-Namen und -Größen
    layers = [
        ("Input", 10, "#87CEEB"),
        *[(f"Hidden {i+1}", units, "#90EE90") for i, units in enumerate(hidden_layers)],
        ("Output", 1, "#FFA07A")
    ]

    # Maximale Breite für Skalierung
    max_units = max(layer[1] for layer in layers)

    for i, (name, units, color) in enumerate(layers):
        # Relative Breite basierend auf Neuronen-Anzahl
        width = (units / max_units) * 4 + 1
        x_center = 2.5
        x_start = x_center - width/2
        x_end = x_center + width/2

        y_pos = len(layers) - i - 1

        # Layer-Rechteck
        fig.add_shape(
            type="rect",
            x0=x_start, y0=y_pos-0.3,
            x1=x_end, y1=y_pos+0.3,
            fillcolor=color,
            line=dict(color="black", width=2),
            opacity=0.8
        )

        # Layer-Text
        fig.add_annotation(
            x=x_center, y=y_pos,
            text=f"{name}<br>({units})",
            showarrow=False,
            font=dict(size=12, color="black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=1
        )

        # Verbindungslinien (außer für letzten Layer)
        if i < len(layers) - 1:
            fig.add_shape(
                type="line",
                x0=x_center, y0=y_pos-0.3,
                x1=x_center, y1=y_pos-0.7,
                line=dict(color="gray", width=2)
            )

    fig.update_layout(
        title="🏗️ Neural Network Architecture",
        xaxis=dict(visible=False, range=[0, 5]),
        yaxis=dict(visible=False, range=[-0.5, len(layers)]),
        height=max(400, len(layers) * 80),
        showlegend=False,
        plot_bgcolor="white"
    )

    return fig

def calculate_parameters(hidden_layers):
    """Berechnet die Anzahl der Parameter"""
    if not hidden_layers:
        return 0

    total_params = 10 * hidden_layers[0] + hidden_layers[0]  # Erste Layer + Bias

    for i in range(len(hidden_layers) - 1):
        total_params += hidden_layers[i] * hidden_layers[i+1] + hidden_layers[i+1]  # Weights + Bias

    total_params += hidden_layers[-1] * 1 + 1  # Output Layer + Bias

    return total_params

def main():
    """🚀 Hauptanwendung"""

    st.set_page_config(
        page_title="🚀 ML Hyperparameter Tuning",
        page_icon="🚀",
        layout="wide"
    )

    st.title("🚀 ML Hyperparameter Tuning & Training")
    st.markdown("**Set Parameters → Start Training → View Results**")

    # Initialize Trainer
    if 'trainer' not in st.session_state:
        st.session_state.trainer = StreamlitMLTrainer()

    # ⚙️ SIDEBAR: Hyperparameter Configuration
    st.sidebar.header("⚙️ Hyperparameter Configuration")

    # Model Architecture - ERWEITERT
    with st.sidebar.expander("🏗️ Model Architecture", expanded=True):
        # Dynamic Layer Management
        if 'layer_config' not in st.session_state:
            st.session_state.layer_config = [128, 64, 32]  # Default

        st.subheader("Layer Verwaltung")

        # Add/Remove Layer Buttons
        col1, col2 = st.columns(2)
        with col1:
            if st.button("➕ Layer hinzufügen"):
                st.session_state.layer_config.append(16)
                st.rerun()

        with col2:
            if st.button("➖ Layer entfernen") and len(st.session_state.layer_config) > 1:
                st.session_state.layer_config.pop()
                st.rerun()

        # Layer Configuration
        hidden_layers = []
        for i, current_units in enumerate(st.session_state.layer_config):
            units = st.slider(
                f"Layer {i+1} Neurons",
                min_value=8,
                max_value=512,
                value=current_units,
                step=8,
                key=f"layer_{i}"
            )
            hidden_layers.append(units)
            # Update session state
            st.session_state.layer_config[i] = units

    # Regularization
    with st.sidebar.expander("🛡️ Regularization", expanded=True):
        dropout_rate = st.slider("Dropout Rate", 0.0, 0.8, 0.3, 0.05)
        l2_reg = st.slider("L2 Regularization", 0.0, 0.1, 0.01, 0.005)

    # Training Parameters
    with st.sidebar.expander("🎯 Training Parameters", expanded=True):
        learning_rate = st.selectbox("Learning Rate", [0.01, 0.005, 0.001, 0.0005], index=2)
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=1)
        max_epochs = st.slider("Max Epochs", 100, 2000, 1000, 100)
        patience = st.slider("Early Stopping Patience", 10, 100, 50, 10)
        test_size = st.slider("Test Size", 0.1, 0.3, 0.2, 0.05)
        validation_size = st.slider("Validation Size", 0.1, 0.3, 0.2, 0.05)

    # Config sammeln
    config = {
        'hidden_layers': hidden_layers,
        'dropout_rate': dropout_rate,
        'l2_reg': l2_reg,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'max_epochs': max_epochs,
        'patience': patience,
        'test_size': test_size,
        'validation_size': validation_size
    }

    # 🏗️ MAIN AREA: Tabs
    tab1, tab2, tab3 = st.tabs(["🏗️ Model Preview", "🚀 Training", "📊 Results"])

    # ============================================================================
    # TAB 1: IMPROVED MODEL PREVIEW
    # ============================================================================
    with tab1:
        st.header("🏗️ Model Architecture Preview")

        col1, col2 = st.columns([3, 1])

        with col1:
            # Verbesserte Architektur-Visualisierung
            fig = create_architecture_visualization(hidden_layers)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Parameter Summary
            total_params = calculate_parameters(hidden_layers)

            st.metric("🔢 Total Parameters", f"{total_params:,}")
            st.metric("📊 Layer Count", len(hidden_layers))
            st.metric("🛡️ Dropout Rate", f"{dropout_rate:.1%}")
            st.metric("⚡ Learning Rate", f"{learning_rate}")
            st.metric("📦 Batch Size", batch_size)

            # Architecture Summary
            st.markdown("**🏗️ Architektur:**")
            st.markdown(f"- Input: 10 Features")
            for i, units in enumerate(hidden_layers):
                st.markdown(f"- Hidden {i+1}: {units} Neurons")
            st.markdown(f"- Output: 1 Neuron")

    # ============================================================================
    # TAB 2: TRAINING INTERFACE
    # ============================================================================
    with tab2:
        st.header("🚀 Training Interface")

        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            if st.button("🚀 Start Training", type="primary",
                        disabled=st.session_state.get('training_active', False)):

                st.session_state.training_active = True
                st.session_state.training_logs = []
                st.session_state.results = None

                with st.spinner("🔄 Training läuft..."):
                    results = st.session_state.trainer.run_complete_training(config)
                    st.session_state.results = results
                    st.session_state.training_active = False

                if 'error' not in results:
                    st.success("✅ Training erfolgreich abgeschlossen!")
                    st.balloons()
                else:
                    st.error(f"❌ Training fehlgeschlagen: {results['error']}")

        with col2:
            if st.button("🧹 Clear Logs"):
                st.session_state.training_logs = []

        with col3:
            if st.button("⏹️ Stop Training"):
                st.session_state.training_active = False

        # Training Status
        if st.session_state.get('training_active', False):
            st.info("🔄 Training läuft... Bitte warten.")

        # Live Training Logs
        st.subheader("📋 Training Logs")

        log_container = st.container()
        with log_container:
            if st.session_state.training_logs:
                for log in reversed(st.session_state.training_logs[-20:]):
                    st.text(log)
            else:
                st.info("🎯 Klicke 'Start Training' um Logs zu sehen...")

    # ============================================================================
    # TAB 3: RESULTS
    # ============================================================================
    with tab3:
        st.header("📊 Training Results Dashboard")

        if not st.session_state.get('results') or 'error' in st.session_state.get('results', {}):
            st.info("🎯 Starte Training um Ergebnisse zu sehen...")
            return

        results = st.session_state.results

        # Key Metrics
        st.subheader("🏆 Performance Metrics")
        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            delta_r2 = f"+{(results['r2'] - 0.49)*100:.1f}%" if results['r2'] > 0.49 else None
            st.metric("🎯 R² Score", f"{results['r2']:.4f}", delta_r2)

        with col2:
            st.metric("📉 MSE", f"{results['mse']:.1f}")

        with col3:
            st.metric("📏 RMSE", f"{results['rmse']:.1f}")

        with col4:
            st.metric("📊 MAE", f"{results['mae']:.1f}")

        with col5:
            st.metric("📈 MAPE", f"{results['mape']:.1f}%")

        # Visualizations
        col1, col2 = st.columns(2)

        with col1:
            # Scatter Plot
            fig_scatter = px.scatter(
                x=results['y_true'], y=results['y_pred'],
                labels={'x': 'Actual Values', 'y': 'Predicted Values'},
                title="🎯 Predictions vs Actual Values"
            )

            min_val, max_val = min(results['y_true']), max(results['y_true'])
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val],
                mode='lines', name='Perfect Prediction',
                line=dict(color='red', dash='dash')
            ))

            st.plotly_chart(fig_scatter, use_container_width=True)

        with col2:
            # Performance Gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=results['r2'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "🎯 R² Performance"},
                delta={'reference': 0.49},
                gauge={
                    'axis': {'range': [0, 1]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 0.4], 'color': "lightgray"},
                        {'range': [0.4, 0.6], 'color': "yellow"},
                        {'range': [0.6, 0.8], 'color': "lightgreen"},
                        {'range': [0.8, 1.0], 'color': "green"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75, 'value': 0.65
                    }
                }
            ))

            st.plotly_chart(fig_gauge, use_container_width=True)

        # Training History
        if st.session_state.trainer.history:
            st.subheader("📈 Training History")

            history_df = pd.DataFrame(st.session_state.trainer.history.history)

            fig_history = go.Figure()
            fig_history.add_trace(go.Scatter(
                x=list(range(len(history_df))),
                y=history_df['loss'],
                mode='lines',
                name='Training Loss',
                line=dict(color='blue')
            ))
            fig_history.add_trace(go.Scatter(
                x=list(range(len(history_df))),
                y=history_df['val_loss'],
                mode='lines',
                name='Validation Loss',
                line=dict(color='red')
            ))

            fig_history.update_layout(
                title="📈 Training & Validation Loss",
                xaxis_title="Epochs",
                yaxis_title="Loss",
                height=400
            )

            st.plotly_chart(fig_history, use_container_width=True)

if __name__ == "__main__":
    main()
