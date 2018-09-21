import numpy as np
from gnuradio import gr
import pmt


class Sylvhronizer(gr.sync_block):  # other base classes are basic_block, decim_block, interp_block
    """Embedded Python Block example - a simple multiply const"""

    def __init__(self, samples_per_symbol=2.5):  # only default arguments here
        """arguments to this function show up as parameters in GRC"""
        gr.sync_block.__init__(
            self,
            name=self.__class__.__name__,   # will show up in GRC
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        # if an attribute with the same name as a parameter is found,
        # a callback is registered (properties work, too).

        self.one_every = samples_per_symbol
        self.phase = float(self.one_every)
        self.previous_sample = None


    def work(self, input_items, output_items):
        """example: multiply with constant"""
        signal = input_items[0]
        
        for k, sample in enumerate(signal):
            self.phase -= 1.0

            if sample != self.previous_sample:
                self.phase = self.one_every / 2.0

            if self.phase < 0.5:
                self.add_item_tag(  0, # Port number
                                    self.nitems_written(0) + k, # Offset
                                    pmt.intern("here"), # Key
                                    pmt.intern('%s' % k))
                self.phase += self.one_every

            self.previous_sample = sample

        output_items[0][:] = input_items[0]
        return len(output_items[0])


class TaggedSamplesPicker(gr.basic_block):

    def __init__(self, tag_key='here'):
        gr.basic_block.__init__(
            self,
            name=self.__class__.__name__,
            in_sig=[np.float32],
            out_sig=[np.float32]
        )
        self.tag_key = tag_key
        self.set_tag_propagation_policy(gr.TPP_CUSTOM)

    def general_work(self, input_items, output_items):
		
        n_output_asked = len(output_items[0])

        tags = self.get_tags_in_window(0, 0, len(input_items[0]))
        
        ks = list(map(lambda tag: tag.offset - self.nitems_read(0), tags))

        if len(ks) <= n_output_asked:
			# It can't produce as many output as asked:
            n_output_produced = len(ks)
            self.consume(0, len(input_items[0]))
        else:
			# It was given more inputs than needed:
			n_output_produced = n_output_asked
			self.consume(0, tags[ks[-1]].offset)

        output_items[0][:n_output_produced] = input_items[0][ks[:n_output_produced]]
        return n_output_produced
