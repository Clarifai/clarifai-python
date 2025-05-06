import glob
import importlib


class ClarifaiModulePageManager(object):
    def __init__(self):
        # List all the available pages.
        page_files = sorted(glob.glob("pages/*.py"))
        self.page_names = [f.replace("pages/", "").replace(".py", "") for f in page_files]

    def get_page_from_query_params(self, qp):
        """
        Args:
          qp: the streamlit query params st.experimental_get_query_params()
        """
        # Get the page from query params or default to 1 from the url.
        page = qp.get("page", [None])[0]
        if page is None:
            page = self.page_names[0]
        # Check that the page number coming in is within the range of pages in the folder.
        if page not in self.page_names:
            raise Exception(
                "Page '%s' is not valid, there is no pages/%s.py file for this page. Valid page names are: %s"
                % (page, page, str(self.page_names))
            )

        return page

    def get_page_names(self):
        return self.page_names

    def render_page(self, page):
        # Since the page re-renders every time the selectbox changes, we'll always have the latest page out
        # of the query params.
        module_str = "pages.%s" % page
        # check if the page exists
        importlib.util.find_spec(module_str)
        if page is None:
            raise Exception("Page %s is was not found." % page)

        current_page = importlib.import_module(module_str)
        current_page.display()
