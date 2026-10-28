FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .
COPY suites ./suites
ENV LLMTESTER_DB=/app/results/llmtester.db
EXPOSE 8000
CMD ["llmtester", "serve", "--host", "0.0.0.0", "--port", "8000"]
