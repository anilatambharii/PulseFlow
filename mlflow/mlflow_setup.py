import os
import mlflow
from mlflow.tracking import MlflowClient
import sys

def setup_mlflow(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "enterprise_mlops_training",
    artifact_location: str = "./mlruns"
):
    """
    Configure MLflow tracking server and create experiment
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the MLflow experiment
        artifact_location: Path to store artifacts
    """
    
    print("=" * 60)
    print("MLflow Setup Configuration")
    print("=" * 60)
    
    # Set tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    print(f"✓ Tracking URI set to: {tracking_uri}")
    
    # Create MLflow client
    client = MlflowClient(tracking_uri=tracking_uri)
    
    # Check if experiment exists
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create new experiment
            experiment_id = mlflow.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location
            )
            print(f"✓ Created new experiment: {experiment_name}")
            print(f"  Experiment ID: {experiment_id}")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Experiment already exists: {experiment_name}")
            print(f"  Experiment ID: {experiment_id}")
            
    except Exception as e:
        print(f"✗ Error creating experiment: {str(e)}")
        sys.exit(1)
    
    # Set the experiment
    mlflow.set_experiment(experiment_name)
    
    # Create artifact directory
    os.makedirs(artifact_location, exist_ok=True)
    print(f"✓ Artifact location: {artifact_location}")
    
    print("=" * 60)
    print("MLflow setup completed successfully!")
    print("=" * 60)
    
    return experiment_id


def verify_mlflow_connection(tracking_uri: str = "http://localhost:5000"):
    """
    Verify connection to MLflow tracking server
    
    Args:
        tracking_uri: MLflow tracking server URI
        
    Returns:
        bool: True if connection successful
    """
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        print(f"✓ Successfully connected to MLflow at {tracking_uri}")
        print(f"  Found {len(experiments)} experiment(s)")
        return True
    except Exception as e:
        print(f"✗ Failed to connect to MLflow: {str(e)}")
        print(f"  Make sure MLflow server is running at {tracking_uri}")
        return False


def list_experiments(tracking_uri: str = "http://localhost:5000"):
    """
    List all experiments in MLflow
    
    Args:
        tracking_uri: MLflow tracking server URI
    """
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        experiments = client.search_experiments()
        
        print("\n" + "=" * 60)
        print("Available MLflow Experiments")
        print("=" * 60)
        
        for exp in experiments:
            print(f"\nName: {exp.name}")
            print(f"  ID: {exp.experiment_id}")
            print(f"  Artifact Location: {exp.artifact_location}")
            print(f"  Lifecycle Stage: {exp.lifecycle_stage}")
            
    except Exception as e:
        print(f"Error listing experiments: {str(e)}")


def create_model_registry(
    model_name: str = "enterprise_mlops_rf_model",
    tracking_uri: str = "http://localhost:5000"
):
    """
    Create a registered model in MLflow Model Registry
    
    Args:
        model_name: Name for the registered model
        tracking_uri: MLflow tracking server URI
    """
    try:
        client = MlflowClient(tracking_uri=tracking_uri)
        
        # Check if model already exists
        try:
            registered_model = client.get_registered_model(model_name)
            print(f"✓ Model already registered: {model_name}")
            print(f"  Description: {registered_model.description}")
        except:
            # Create new registered model
            client.create_registered_model(
                name=model_name,
                description="Enterprise MLOps Random Forest Regression Model"
            )
            print(f"✓ Created new registered model: {model_name}")
            
    except Exception as e:
        print(f"Error creating model registry: {str(e)}")


def log_sample_run(
    tracking_uri: str = "http://localhost:5000",
    experiment_name: str = "enterprise_mlops_training"
):
    """
    Log a sample run to verify MLflow is working
    
    Args:
        tracking_uri: MLflow tracking server URI
        experiment_name: Name of the experiment
    """
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    
    with mlflow.start_run(run_name="sample_test_run"):
        # Log sample parameters
        mlflow.log_param("test_param", "test_value")
        mlflow.log_param("n_estimators", 100)
        
        # Log sample metrics
        mlflow.log_metric("test_metric", 0.95)
        mlflow.log_metric("mse", 0.05)
        
        print("✓ Sample run logged successfully")


def main():
    """Main setup function"""
    
    # Configuration
    TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    EXPERIMENT_NAME = "enterprise_mlops_training"
    ARTIFACT_LOCATION = "./mlruns"
    MODEL_NAME = "enterprise_mlops_rf_model"
    
    print("\nStarting MLflow setup...\n")
    
    # Verify connection
    print("Step 1: Verifying MLflow connection...")
    if not verify_mlflow_connection(TRACKING_URI):
        print("\n⚠ Warning: MLflow server not accessible")
        print("  Start MLflow server with: mlflow server --host 0.0.0.0 --port 5000")
        print("  Or run: mlflow ui")
        return
    
    # Setup experiment
    print("\nStep 2: Setting up experiment...")
    experiment_id = setup_mlflow(
        tracking_uri=TRACKING_URI,
        experiment_name=EXPERIMENT_NAME,
        artifact_location=ARTIFACT_LOCATION
    )
    
    # Create model registry
    print("\nStep 3: Creating model registry...")
    create_model_registry(
        model_name=MODEL_NAME,
        tracking_uri=TRACKING_URI
    )
    
    # Log sample run
    print("\nStep 4: Logging sample run...")
    log_sample_run(
        tracking_uri=TRACKING_URI,
        experiment_name=EXPERIMENT_NAME
    )
    
    # List experiments
    list_experiments(TRACKING_URI)
    
    print("\n" + "=" * 60)
    print("✓ MLflow setup completed successfully!")
    print("=" * 60)
    print(f"\nAccess MLflow UI at: {TRACKING_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
    print(f"Model Registry: {MODEL_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()
