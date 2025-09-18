from gold import app as application

# This block allows Heroku to run the app using the 'web: gunicorn app:application' command.
if __name__ == '__main__':
    application.run()
