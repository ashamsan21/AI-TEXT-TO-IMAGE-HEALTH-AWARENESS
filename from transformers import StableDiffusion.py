from transformers import StableDiffusionPipeline, Trainer, TrainingArguments

def finetune_model(train_data):
    # Load pre-trained model
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4")

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=train_data,  # In a real scenario, use a separate validation set
    )

    # Start training
    trainer.train()

if __name__ == "__main__":
    # Assume train_data is prepared using data_preparation script
    from src.data_preparation import prepare_data
    train_data = prepare_data()

    # Fine-tune the model
    finetune_model(train_data)
