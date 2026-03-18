FROM python:3.12-slim
WORKDIR /app

# TODO: install the alpha-beta-CROWN worker dependencies and copy the adapter package.
CMD ["python", "-c", "print('alpha-beta-crown worker scaffold')"]

