from pathlib import Path


class RealESRGANAdapter:
    def __init__(self, repo_path: Path):
        self.repo_path = repo_path

    def validate(self) -> bool:
        return self.repo_path.exists() and self.repo_path.is_dir()

    def load(self, model_name: str):
        if not self.validate():
            raise FileNotFoundError('Real-ESRGAN repository path is invalid')
        return {'model_name': model_name, 'repo_path': str(self.repo_path)}
